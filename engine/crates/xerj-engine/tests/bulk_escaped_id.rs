//! Issue #954: `_bulk` truncated an action-line `_id` at an escaped quote.
//!
//! The fast path that extracts `_id`/`_index`/`op_type` from an action line
//! scanned for the next raw `"` without unescaping, so `{"_id":"a\"b"}` parsed
//! as id `a\` and the line's remainder desynced — two documents with distinct
//! ids collapsed onto one, and the second write either overwrote the first or
//! 409'd as a duplicate create. The fix bails out of the fast path whenever
//! the action object contains a backslash (any escape) and lets the serde_json
//! fallback parse the line correctly.

use serde_json::json;
use tempfile::TempDir;
use xerj_common::config::Config;
use xerj_common::types::{FieldConfig, FieldType, Schema};
use xerj_engine::bulk::process_bulk;
use xerj_engine::Engine;

fn make_engine(dir: &TempDir) -> Engine {
    let mut config = Config::default();
    config.server.data_dir = dir.path().to_str().unwrap().to_string();
    Engine::new(config).expect("engine::new")
}

fn seed(engine: &Engine, name: &str) {
    let mut schema = Schema::empty();
    schema
        .fields
        .push(FieldConfig::new("body", FieldType::Text));
    engine.create_index(name, schema).expect("create index");
}

/// Distinct ids that differ only inside an escaped segment must stay distinct:
/// all items index (no duplicate-create 409), and a GET-by-id finds each body
/// under its own id. FAIL-BEFORE: the fast path truncates both quote-escaped
/// ids to `a\`, so the two collapse and one body shadows the other; the
/// `\uXXXX` id is stored as its literal 6 escape characters (backslash
/// u 0 0 e 9) instead of the character it names.
#[tokio::test]
async fn escaped_quote_ids_stay_distinct() {
    let dir = TempDir::new().unwrap();
    let engine = make_engine(&dir);
    seed(&engine, "t");

    // Raw NDJSON. JSON-unescaped, the three ids are `a"b`, `a"c` and `é`
    // (the last via a \uXXXX escape, which the fast path used to store
    // literally).
    let body = concat!(
        r#"{"index":{"_index":"t","_id":"a\"b"}}"#,
        "\n",
        r#"{"body":"first"}"#,
        "\n",
        r#"{"index":{"_index":"t","_id":"a\"c"}}"#,
        "\n",
        r#"{"body":"second"}"#,
        "\n",
        r#"{"index":{"_index":"t","_id":"\u00e9"}}"#,
        "\n",
        r#"{"body":"third"}"#,
        "\n",
    );
    let result = process_bulk(&engine, None, body).await;
    assert!(!result.errors, "bulk must not error: {:?}", result.items);
    assert_eq!(result.items.len(), 3);
    // The response must echo the FULL unescaped id, never the truncation.
    assert_eq!(result.items[0].id, "a\"b");
    assert_eq!(result.items[1].id, "a\"c");
    assert_eq!(result.items[2].id, "é");

    let idx = engine.get_index("t").expect("get index");
    // Rust-side: "a\"b" is a"b and "a\"c" is a"c — the JSON-unescaped ids.
    for (id, body_text) in [("a\"b", "first"), ("a\"c", "second"), ("é", "third")] {
        let doc = idx
            .get_document(id)
            .await
            .expect("lookup")
            .unwrap_or_else(|| panic!("document {id:?} must exist"));
        assert_eq!(
            doc.get("body").and_then(|b| b.as_str()),
            Some(body_text),
            "id {id:?} must hold its own body"
        );
    }
}

/// The escape opt-out must not change behaviour for ids WITHOUT escapes
/// (the fast path stays exact for the canonical case), and an invalid escape
/// sequence must surface as a per-item 400, never as a silent truncation.
#[tokio::test]
async fn plain_ids_unchanged_and_bad_escape_is_a_per_item_error() {
    let dir = TempDir::new().unwrap();
    let engine = make_engine(&dir);
    seed(&engine, "t");

    let body = concat!(
        r#"{"index":{"_index":"t","_id":"plain"}}"#,
        "\n",
        r#"{"body":"ok"}"#,
        "\n",
    );
    let result = process_bulk(&engine, None, body).await;
    assert!(
        !result.errors,
        "plain ids must keep working: {:?}",
        result.items
    );
    let idx = engine.get_index("t").expect("get index");
    assert!(idx.get_document("plain").await.expect("lookup").is_some());

    // `\x` is not a valid JSON escape: the serde fallback must reject the
    // action line as a 400 item with the parse error, not truncate `_id`.
    let body = concat!(
        r#"{"index":{"_index":"t","_id":"bad\x22id"}}"#,
        "\n",
        r#"{"body":"never"}"#,
        "\n",
    );
    let result = process_bulk(&engine, None, body).await;
    assert_eq!(result.items.len(), 1, "one action, one item");
    assert_eq!(result.items[0].status, 400, "invalid escape must 400");
    assert!(result.items[0].error.is_some());
    let all = idx
        .search(
            &xerj_query::parse_request(&json!({
                "query": { "match": { "body": "never" } }, "size": 10
            }))
            .expect("parse"),
        )
        .await
        .expect("search");
    assert_eq!(all.total.value, 0, "the rejected item must not be indexed");
}
