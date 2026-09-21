//! Issue #932 — a `fields` clause must resolve against the intact stored
//! source even when an explicit `_source` includes list does not contain the
//! requested field, and the wire `_source` must stay narrowed to that list.
//!
//! The engine applies an explicit `_source` includes/excludes projection to
//! `hit.source` BEFORE the response layer runs, and `fields` /
//! `docvalue_fields` resolve their values out of that same narrowed source —
//! so `{"_source": ["title"], "fields": ["body"]}` legally omitted
//! `fields.body`: HTTP 200, no warning, nothing. The identical clause works
//! with no `_source` filter and with `_source: false` (which deliberately
//! keeps the raw source for exactly this resolution). Elasticsearch treats
//! the two clauses as independent: `fields` reads the stored `_source`,
//! source filtering only selects "what fields of the source are returned".
//!
//! The fix is the #310 approach applied to the explicit-filter path: ask the
//! engine for the intact source when the caller's own includes/excludes drop
//! a value-bearing name, then re-apply the caller's filter at every `_source`
//! emission site. Every assertion below is therefore paired — the POSITIVE
//! half (the value comes back) is the fix; the NEGATIVE half (`_source`
//! carries exactly the caller's projection, never the pierced field) is what
//! keeps the pierce honest.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;
use xerj_common::types::{FieldConfig, FieldType, Schema};

/// Long bodies on purpose: the `_savings` note has a print floor (a saving
/// smaller than [`SAVINGS_REPORT_FLOOR_RATIO`] times the block costs is not
/// printed), and the last test uses the note's PRESENCE as its oracle for
/// "this request did not pierce".
const DOCS: [(&str, &str, &str, i64); 4] = [
    (
        "1",
        "Trial results one",
        "vitamin d supplementation improved bone density in the treatment group, \
         with sustained gains observed across every follow-up visit and no \
         attrition in the cohort assigned to the active arm of the study",
        11,
    ),
    (
        "2",
        "Trial results two",
        "the placebo group showed no change over the study period, matching the \
         pre-registered expectation for the control arm and confirming that the \
         measured effect in the treated cohort is attributable to the invention",
        7,
    ),
    (
        "3",
        "Trial results three",
        "secondary endpoints favoured the treatment group on every measure that \
         the protocol designated as exploratory, and the direction of effect was \
         consistent with the primary endpoint throughout the observation window",
        5,
    ),
    (
        "4",
        "Trial results four",
        "no serious adverse events were recorded in either arm during the entire \
         study, and the safety monitoring board recommended continuation without \
         modification at each of its scheduled interim reviews",
        2,
    ),
];

/// The seeded `body` value: the base sentence repeated, so the bytes an
/// includes list omits clear the savings print floor.
fn full_body(base: &str) -> String {
    format!("{base} {base} {base}")
}

async fn seeded_app() -> (axum::Router, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = xerj_common::config::Config::default();
    config.server.data_dir = dir.path().to_string_lossy().into_owned();
    config.storage.wal_sync = xerj_common::config::WalSync::Async;
    let metrics = xerj_common::metrics::Metrics::new().expect("metrics");
    let engine = xerj_engine::Engine::new(config.clone()).expect("engine");
    let state = xerj_api::state::AppState::new(config, engine, metrics);

    let mut schema = Schema::empty();
    schema
        .add_field(FieldConfig::new("title", FieldType::Text))
        .expect("title field");
    schema
        .add_field(FieldConfig::new("body", FieldType::Text))
        .expect("body field");
    schema
        .add_field(FieldConfig::new("cat", FieldType::Keyword))
        .expect("cat field");
    schema
        .add_field(FieldConfig::new("n", FieldType::Long))
        .expect("n field");
    state.engine.create_index("docs", schema).expect("create");
    let idx = state.engine.get_index("docs").expect("get index");
    for (id, title, body, n) in DOCS {
        idx.index_document(
            Some(id.into()),
            json!({
                "title": title,
                "body": full_body(body),
                "cat": format!("cat-{id}"),
                "n": n
            }),
        )
        .await
        .expect("index document");
    }
    idx.refresh().await.expect("refresh");

    (xerj_api::router::build_es_compat_router(state), dir)
}

async fn post(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

fn hits(response: &Value) -> &Vec<Value> {
    response["hits"]["hits"]
        .as_array()
        .unwrap_or_else(|| panic!("no hits.hits array: {response}"))
}

/// The issue's repro table, row by row. Rows 1-3 already worked on main and
/// must keep working; rows 4-5 are the bug (`fields` silently dropped for a
/// text AND a keyword field the includes list omits).
#[tokio::test]
async fn fields_survives_every_source_includes_combination() {
    let (app, _dir) = seeded_app().await;

    // Row 1: no `_source` filter — `fields.body` returned.
    let (status, resp) = post(
        &app,
        "/docs/_search",
        json!({ "query": { "match_all": {} }, "size": 4, "fields": ["body"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["body"],
            json!([full_body(DOCS[i].2)]),
            "row 1 (no _source): fields.body: {hit}"
        );
    }

    // Row 2: `_source: false` — same clause, still returned.
    let (status, resp) = post(
        &app,
        "/docs/_search",
        json!({ "query": { "match_all": {} }, "size": 4, "_source": false, "fields": ["body"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["body"],
            json!([full_body(DOCS[i].2)]),
            "row 2 (_source false): fields.body: {hit}"
        );
        assert!(
            hit.get("_source").is_none(),
            "row 2: _source suppressed: {hit}"
        );
    }

    // Row 3: includes list CONTAINS the requested field — works, and must
    // keep working (a covered name does not pierce).
    let (status, resp) = post(
        &app,
        "/docs/_search",
        json!({ "query": { "match_all": {} }, "size": 4, "_source": ["title"], "fields": ["title"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["title"],
            json!([DOCS[i].1]),
            "row 3 (covered name): fields.title: {hit}"
        );
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i].1 }),
            "row 3: _source stays narrowed to the includes list: {hit}"
        );
    }

    // Row 4: `_source: ["title"]`, `fields: ["body"]` — the bug. `fields.body`
    // must come back AND `_source` must carry exactly the caller's projection.
    let (status, resp) = post(
        &app,
        "/docs/_search",
        json!({ "query": { "match_all": {} }, "size": 4, "_source": ["title"], "fields": ["body"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["body"],
            json!([full_body(DOCS[i].2)]),
            "row 4 (#932): fields.body must pierce the explicit includes: {hit}"
        );
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i].1 }),
            "row 4 (#932): _source must stay narrowed to title: {hit}"
        );
    }

    // Row 5: same for a keyword field — not specific to text.
    let (status, resp) = post(
        &app,
        "/docs/_search",
        json!({ "query": { "match_all": {} }, "size": 4, "_source": ["title"], "fields": ["cat"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["cat"],
            json!([format!("cat-{}", DOCS[i].0)]),
            "row 5 (#932): fields.cat (keyword) must pierce too: {hit}"
        );
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i].1 }),
            "row 5 (#932): _source must stay narrowed to title: {hit}"
        );
    }
}

/// The object spelling and both URL spellings of the same request.
#[tokio::test]
async fn every_spelling_of_the_request_pierces() {
    let (app, _dir) = seeded_app().await;

    // `_source: {"includes": ["title"]}`.
    let (status, resp) = post(
        &app,
        "/docs/_search",
        json!({
            "query": { "match_all": {} }, "size": 4,
            "_source": { "includes": ["title"] }, "fields": ["body"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["body"],
            json!([full_body(DOCS[i].2)]),
            "object spelling: fields.body: {hit}"
        );
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i].1 }),
            "object spelling: _source narrowed: {hit}"
        );
    }

    // `?_source_includes=title` with a body `fields` clause.
    let (status, resp) = post(
        &app,
        "/docs/_search?_source_includes=title",
        json!({ "query": { "match_all": {} }, "size": 4, "fields": ["body"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["body"],
            json!([full_body(DOCS[i].2)]),
            "URL _source_includes spelling: fields.body: {hit}"
        );
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i].1 }),
            "URL _source_includes spelling: _source narrowed: {hit}"
        );
    }

    // `?_source=title` + `?fields=body`, both via URL.
    let (status, resp) = post(
        &app,
        "/docs/_search?_source=title&fields=body",
        json!({ "query": { "match_all": {} }, "size": 4 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["body"],
            json!([full_body(DOCS[i].2)]),
            "URL _source= spelling: fields.body: {hit}"
        );
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i].1 }),
            "URL _source= spelling: _source narrowed: {hit}"
        );
    }
}

/// `docvalue_fields` resolves out of the same narrowed source — the identical
/// silent drop for a non-text field.
#[tokio::test]
async fn docvalue_fields_pierce_the_includes_list() {
    let (app, _dir) = seeded_app().await;

    let (status, resp) = post(
        &app,
        "/docs/_search",
        json!({
            "query": { "match_all": {} }, "size": 4,
            "_source": ["title"], "docvalue_fields": ["n"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["n"],
            json!([DOCS[i].3]),
            "#932: docvalue_fields.n must pierce the includes: {hit}"
        );
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i].1 }),
            "#932: _source must stay narrowed to title: {hit}"
        );
    }
}

/// An excludes-only filter (`_source: {"excludes": ["title"]}`) drops every
/// OTHER field from nothing — `fields: ["title"]` names exactly the excluded
/// field, the sharpest form of the bug.
#[tokio::test]
async fn an_excluded_field_name_still_resolves_through_fields() {
    let (app, _dir) = seeded_app().await;

    let (status, resp) = post(
        &app,
        "/docs/_search",
        json!({
            "query": { "match_all": {} }, "size": 4,
            "_source": { "excludes": ["title"] }, "fields": ["title"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{resp}");
    for (i, hit) in hits(&resp).iter().enumerate() {
        assert_eq!(
            hit["fields"]["title"],
            json!([DOCS[i].1]),
            "#932: fields.title names the excluded field and must resolve: {hit}"
        );
        let src = hit["_source"].as_object().expect("_source object");
        assert!(!src.contains_key("title"), "excludes still applies: {hit}");
        assert!(src.contains_key("body"), "unexcluded fields stay: {hit}");
    }
}

/// #932 emission site 3: a scroll opened by a piercing request must NOT leak
/// the pierced field on the CONTINUATION pages — the scroll snapshot is
/// re-narrowed before the context is stored.
#[tokio::test]
async fn scroll_continuation_stays_narrowed_after_a_pierce() {
    let (app, _dir) = seeded_app().await;

    let (status, page1) = post(
        &app,
        "/docs/_search?scroll=5m",
        json!({
            "query": { "match_all": {} }, "size": 2,
            "_source": ["title"], "fields": ["body"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{page1}");
    for (i, hit) in hits(&page1).iter().enumerate() {
        assert_eq!(
            hit["fields"]["body"],
            json!([full_body(DOCS[i].2)]),
            "page 1 resolves fields.body through the pierce: {hit}"
        );
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i].1 }),
            "page 1 _source stays narrowed: {hit}"
        );
    }
    let sid = page1["_scroll_id"].as_str().expect("scroll id").to_string();

    let (status, page2) = post(
        &app,
        "/_search/scroll",
        json!({ "scroll": "5m", "scroll_id": sid }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{page2}");
    for (i, hit) in hits(&page2).iter().enumerate() {
        assert_eq!(
            hit["_source"],
            json!({ "title": DOCS[i + 2].1 }),
            "#932 site 3: continuation pages must not leak the pierced body: {hit}"
        );
    }
}

/// The pierce is an instrumentation trade, not a free fix: `Enabled(true)` is
/// not a measured shape in the engine, so a PIERCED request loses its
/// `_savings` note (the same trade #310's default-projection pierce makes) —
/// while a filter that already covers every named field must NOT pierce, and
/// so keeps the note it has always had. That asymmetry is the proof that the
/// coverage check keeps unaffected requests on the un-pierced path.
#[tokio::test]
async fn a_covered_name_keeps_the_savings_note_a_pierced_one_trades_it() {
    let (app, _dir) = seeded_app().await;

    // Covered (`title` is in the includes list): no pierce, `_savings` intact.
    let (status, covered) = post(
        &app,
        "/docs/_search",
        json!({ "query": { "match_all": {} }, "size": 4, "_source": ["title"], "fields": ["title"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{covered}");
    assert!(
        covered["_savings"].is_object(),
        "a covered name does not pierce; the default savings note stays: {covered}"
    );

    // Pierced (`body` is dropped by the includes list): the note is the cost.
    let (status, pierced) = post(
        &app,
        "/docs/_search",
        json!({ "query": { "match_all": {} }, "size": 4, "_source": ["title"], "fields": ["body"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{pierced}");
    assert!(
        pierced.get("_savings").is_none(),
        "a pierced request asks the engine for the intact source — nothing was \
         withheld, and no note may claim otherwise: {pierced}"
    );
    // …and the value the caller asked for is there.
    assert_eq!(
        pierced["hits"]["hits"][0]["fields"]["body"],
        json!([full_body(DOCS[0].2)]),
        "{pierced}"
    );
}
