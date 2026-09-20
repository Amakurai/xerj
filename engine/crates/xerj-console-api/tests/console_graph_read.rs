//! Session-authorized console graph reads (issue #936).
//!
//! The console holds a passkey session, not an engine API key, and the
//! data plane's `/_graph/{brain}/*` routes authenticate only
//! Authorization-header credentials — so on an auth-enabled engine (the
//! default) a signed-in operator could not read any brain while a
//! share-link guest could. `xerj-console-api::graph` is the console's own
//! role-gated read path. These tests pin its boundary:
//!
//! - the operator tier (`owner`/`admin`) reads brains through the console
//!   endpoints — listing, `ego` with node hydration, `overview`, the
//!   brain-pinned edges and nodes searches;
//! - a disallowed role (`viewer`) and a nonexistent brain are
//!   **indistinguishable** on every per-brain endpoint (no existence
//!   oracle), and no brain content reaches a refused session;
//! - the `brains` listing is a flat role gate (403, no per-brain fact);
//! - no session at all is a plain 401 on every route.
//!
//! The data plane's own boundary (`/_graph/*` still requires an API key;
//! the generic data-sources proxy still refuses the reserved namespace,
//! RC10 B1) is pinned where it lives: `xerj-api`'s
//! `brain_is_a_security_boundary.rs` and `console_cannot_read_a_brain.rs`.

use axum::{body::Body, http::Request, Router};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;
use xerj_common::config::Config;
use xerj_common::types::{FieldConfig, FieldType, Schema};
use xerj_console_api::{
    auth::{sessions, store},
    state::ClusterMode,
    xerj_console_router, ConsoleState,
};
use xerj_engine::Engine;

/// Content that must never reach a session that may not read brains.
const EDGE_SECRET: &str = "alice-private-fact-9f83c1";

const T0: i64 = 1_753_600_000_000; // valid_at / created_at of the fixture edges
const T_INVALID: i64 = 1_753_650_000_000; // invalid_at of the retired edge

struct TestApp {
    router: Router,
    owner_cookie: String,
    viewer_cookie: String,
    _dir: TempDir,
}

/// The §2.1 edges-index schema (the subset the traversal and the hydrations
/// read): keyword columns for the hop, numeric timestamps for the
/// bi-temporal cut, keyword aggregates for `overview`.
fn edges_schema() -> Schema {
    let mut s = Schema::empty();
    fn add(s: &mut Schema, name: &str, t: FieldType) {
        s.add_field(FieldConfig::new(name, t)).expect("add field");
    }
    for f in ["edge_id", "src", "dst", "type", "detector"] {
        add(&mut s, f, FieldType::Keyword);
    }
    for f in ["valid_at", "invalid_at", "created_at", "expired_at"] {
        add(&mut s, f, FieldType::Date);
    }
    for f in ["weight", "confidence"] {
        add(&mut s, f, FieldType::Double);
    }
    add(&mut s, "schema_version", FieldType::Long);
    s
}

async fn add_session_user(state: &ConsoleState, engine: &Engine, id: &str, role: &str) -> String {
    let user = store::User {
        id: id.to_string(),
        email: format!("{id}@example.com"),
        display_name: id.to_string(),
        role: role.to_string(),
        status: store::UserStatus::Active,
        created_at: xerj_console_api::time::now_iso(),
        last_seen_at: Some(xerj_console_api::time::now_iso()),
    };
    store::upsert_user(engine, &user).await.unwrap();
    let (_s, signed) = sessions::mint_session(state, &user.id, "passkey", None, None)
        .await
        .unwrap();
    format!("xerj_session={signed}")
}

/// Boot the console over one brain, exactly as `xerj brain` would leave the
/// node: an edges index with a §2.5 meta doc and three edges (one retired),
/// plus the `ax-docs` nodes index the meta doc names.
async fn boot_with_brain() -> TestApp {
    let dir = TempDir::new().unwrap();
    let mut cfg = Config::default();
    cfg.server.data_dir = dir.path().to_str().unwrap().to_string();
    let engine = Engine::new(cfg).expect("engine");
    let outcome = xerj_console_api::bootstrap::run(&engine, dir.path(), "http://localhost:9200")
        .await
        .unwrap();
    let state = ConsoleState::new(
        engine.clone(),
        "local".into(),
        outcome.master_key,
        ClusterMode::Standalone,
    );

    let owner_cookie = add_session_user(&state, &engine, "owner-test", "owner").await;
    let viewer_cookie = add_session_user(&state, &engine, "viewer-test", "viewer").await;

    engine
        .create_index(".xerj-memory-casefile-edges", edges_schema())
        .expect("create edges index");
    let edges = engine.get_index(".xerj-memory-casefile-edges").unwrap();

    // §2.5 meta doc: this brain's nodes index is the ordinary dataset index.
    edges
        .index_document(
            Some("__xerj-brain-meta".into()),
            json!({
                "meta_version": 1,
                "brain": "casefile",
                "nodes_index": "ax-docs",
                "created_at": T0,
            }),
        )
        .await
        .unwrap();

    let edge = |id: &str, src: &str, dst: &str, t: &str, extra: Value| {
        let mut doc = json!({
            "edge_id": id,
            "src": src,
            "dst": dst,
            "type": t,
            "weight": 1.0,
            "valid_at": T0,
            "created_at": T0,
            "detector": "same_dir@1",
            "confidence": 1.0,
            "schema_version": 1,
            "evidence": { "quote": EDGE_SECRET, "source": "inbox/01.eml" },
        });
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                doc[k] = v.clone();
            }
        }
        (Some(id.to_string()), doc)
    };
    edges
        .index_documents_batched(vec![
            edge("e1", "file-01", "file-02", "same_dir", json!({})),
            edge("e2", "file-01", "file-03", "mdlink", json!({})),
            edge(
                "e3",
                "file-02",
                "file-03",
                "pathcite",
                json!({ "invalid_at": T_INVALID, "expired_at": T_INVALID }),
            ),
        ])
        .await;

    engine
        .create_index("ax-docs", Schema::empty())
        .expect("create nodes index");
    let nodes = engine.get_index("ax-docs").unwrap();
    nodes
        .index_documents_batched(vec![
            (
                Some("file-01".into()),
                json!({ "title": "01.eml", "text": "the first message", "ax_path": "inbox/01.eml" }),
            ),
            (
                Some("file-02".into()),
                json!({ "title": "02.eml", "text": "the second message", "ax_path": "inbox/02.eml" }),
            ),
            (
                Some("file-03".into()),
                json!({ "title": "03.eml", "text": "the third message", "ax_path": "inbox/03.eml" }),
            ),
        ])
        .await;

    let router = xerj_console_router(state);
    TestApp {
        router,
        owner_cookie,
        viewer_cookie,
        _dir: dir,
    }
}

async fn body_json(resp: axum::response::Response) -> (axum::http::StatusCode, Value, String) {
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let v = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, v, text)
}

fn req(method: &str, path: &str, cookie: Option<&str>, body: Option<Value>) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(path);
    if let Some(c) = cookie {
        b = b.header("cookie", c);
    }
    let body = match body {
        Some(v) => {
            b = b.header("content-type", "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    b.body(body).unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────

/// The full product path the issue reports broken: the operator lists the
/// node's brains, walks one node's neighborhood with hydrated titles, reads
/// the brain-level stats, and the dashboard's edges/nodes searches answer.
#[tokio::test]
async fn owner_reads_the_graph() {
    let app = boot_with_brain().await;

    // Discovery: the listing replaces `_cat/indices/.xerj-memory-*` + the
    // per-brain meta-doc reads, both of which 401 for a session.
    let (status, body, _) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "GET",
                "/_xerj-console/api/v1/graph/brains",
                Some(&app.owner_cookie),
                None,
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "brains listing: {body}");
    let brains = body["data"]["brains"].as_array().expect("brains array");
    assert_eq!(brains.len(), 1, "one brain: {body}");
    assert_eq!(brains[0]["name"], "casefile");
    assert_eq!(brains[0]["nodes_index"], "ax-docs");

    // ego: 2 live edges at now (e3 is retired), neighbors, hydrated titles.
    let (status, body, _) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "GET",
                "/_xerj-console/api/v1/graph/casefile/ego?node=file-01&hops=1&direction=both&include_nodes=true",
                Some(&app.owner_cookie),
                None,
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "ego: {body}");
    assert_eq!(body["brain"], "casefile");
    assert_eq!(body["node"], "file-01", "single-seed keeps the `node` key");
    let edges = body["edges"].as_array().expect("edges");
    assert_eq!(edges.len(), 2, "the retired edge is excluded: {body}");
    assert!(edges.iter().all(|e| e["invalid_at"].is_null()));
    assert!(
        body["nodes"]["file-02"]["title"] == "02.eml",
        "node summaries hydrate: {body}"
    );
    assert_eq!(body["not_shown"]["dangling_nodes"], 0);

    // include_expired brings the retirement back (the ledger's struck rows).
    let (status, body, _) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "GET",
                "/_xerj-console/api/v1/graph/casefile/ego?node=file-02&include_expired=true",
                Some(&app.owner_cookie),
                None,
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(body["edges"].as_array().unwrap().len(), 2);

    // overview: totals, live/invalidated split, hubs, notes total.
    let (status, body, _) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "GET",
                "/_xerj-console/api/v1/graph/casefile/overview?top=10&histogram_interval=day",
                Some(&app.owner_cookie),
                None,
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "overview: {body}");
    assert_eq!(body["exists"], true);
    assert_eq!(body["edges"]["total"], 3);
    assert_eq!(body["edges"]["live"], 2);
    assert_eq!(body["edges"]["invalidated"], 1);
    assert_eq!(body["nodes"]["total"], 3, "notes counted on ax-docs");
    assert_eq!(body["nodes_index"], "ax-docs");
    assert_eq!(body["embedder"], "lexical-feature-hash");
    assert!(
        !body["hubs"]["out"].as_array().expect("hubs out").is_empty(),
        "hub ids from the live slice"
    );

    // The dashboard's edges search (recent retirements) — brain-pinned.
    let (status, body, _) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "POST",
                "/_xerj-console/api/v1/graph/casefile/edges/_search",
                Some(&app.owner_cookie),
                Some(json!({
                    "query": { "exists": { "field": "invalid_at" } },
                    "sort": [{ "invalid_at": { "order": "desc" } }],
                    "size": 3,
                })),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "edges search: {body}");
    let hits = body["hits"]["hits"].as_array().expect("hits");
    assert_eq!(hits.len(), 1, "exactly the retired edge: {body}");
    assert_eq!(hits[0]["_id"], "e3");

    // The dashboard's nodes search (notes tally / FIND).
    let (status, body, _) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "POST",
                "/_xerj-console/api/v1/graph/casefile/nodes/_search",
                Some(&app.owner_cookie),
                Some(json!({
                    "size": 0,
                    "track_total_hits": true,
                    "aggs": { "formats": { "terms": { "field": "ax_format", "size": 12 } } },
                })),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 200, "nodes search: {body}");
    assert_eq!(body["hits"]["total"]["value"], 3);
}

/// A viewer's refusal and a nonexistent brain are byte-identical on every
/// per-brain endpoint — the same no-existence-oracle discipline the search
/// proxy pinned — and no brain content reaches the refused session.
#[tokio::test]
async fn refusal_is_no_existence_oracle() {
    let app = boot_with_brain().await;
    const GHOST: &str = "no-such-brain";

    async fn viewer_call(
        app: &TestApp,
        method: &str,
        brain: &str,
        tail: &str,
        body: Option<Value>,
    ) -> (axum::http::StatusCode, Value, String) {
        let path = format!("/_xerj-console/api/v1/graph/{brain}{tail}");
        body_json(
            app.router
                .clone()
                .oneshot(req(method, &path, Some(&app.viewer_cookie), body))
                .await
                .unwrap(),
        )
        .await
    }

    let cases: &[(&str, &str, &str)] = &[
        ("GET", "/ego?node=file-01", ""),
        ("GET", "/overview", ""),
        ("POST", "/edges/_search", r#"{"query":{"match_all":{}}}"#),
        ("POST", "/nodes/_search", r#"{"query":{"match_all":{}}}"#),
    ];
    for (method, tail, raw_body) in cases {
        let body = if raw_body.is_empty() {
            None
        } else {
            serde_json::from_str(raw_body).ok()
        };
        let (real_status, real_body, real_text) =
            viewer_call(&app, method, "casefile", tail, body.clone()).await;
        let (ghost_status, _ghost_body, ghost_text) =
            viewer_call(&app, method, GHOST, tail, body).await;
        assert_eq!(real_status, ghost_status, "{method} {tail}: status differs");
        assert_eq!(real_status, 404, "{method} {tail}: refusal must 404");
        assert_eq!(
            real_text.replace("casefile", GHOST),
            ghost_text,
            "{method} {tail}: existing vs missing brain must be indistinguishable"
        );
        assert!(
            !real_text.contains(EDGE_SECRET),
            "{method} {tail}: brain content leaked to a viewer"
        );
        assert!(
            real_body["error"]["type"] == "graph_error" || real_body["exists"] == false,
            "{method} {tail}: unexpected refusal body {real_body}"
        );
    }

    // The listing is a flat role gate: 403, and it names no brain.
    let (status, body, text) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "GET",
                "/_xerj-console/api/v1/graph/brains",
                Some(&app.viewer_cookie),
                None,
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 403, "viewer listing: {body}");
    assert!(
        !text.contains("casefile"),
        "listing leaked a brain name: {text}"
    );
}

/// The admin role may read; the editor role may not (the rule is the
/// operator tier, not one role name).
#[tokio::test]
async fn the_operator_tier_is_owner_and_admin() {
    let user = |role: &str| store::User {
        id: "u".into(),
        email: "u@example.com".into(),
        display_name: "u".into(),
        role: role.into(),
        status: store::UserStatus::Active,
        created_at: xerj_console_api::time::now_iso(),
        last_seen_at: None,
    };
    assert!(xerj_console_api::graph::may_read_brains(&user("owner")));
    assert!(xerj_console_api::graph::may_read_brains(&user("admin")));
    assert!(!xerj_console_api::graph::may_read_brains(&user("editor")));
    assert!(!xerj_console_api::graph::may_read_brains(&user("viewer")));
}

/// No session at all is a plain 401 on every graph route (the AuthSession
/// gate runs before anything brain-related).
#[tokio::test]
async fn no_session_is_unauthorized() {
    let app = boot_with_brain().await;
    let routes: &[(&str, &str, Option<Value>)] = &[
        ("GET", "/_xerj-console/api/v1/graph/brains", None),
        (
            "GET",
            "/_xerj-console/api/v1/graph/casefile/ego?node=file-01",
            None,
        ),
        ("GET", "/_xerj-console/api/v1/graph/casefile/overview", None),
        (
            "POST",
            "/_xerj-console/api/v1/graph/casefile/edges/_search",
            Some(json!({"query":{"match_all":{}}})),
        ),
        (
            "POST",
            "/_xerj-console/api/v1/graph/casefile/nodes/_search",
            Some(json!({"query":{"match_all":{}}})),
        ),
    ];
    for (method, path, body) in routes {
        let (status, _, _) = body_json(
            app.router
                .clone()
                .oneshot(req(method, path, None, body.clone()))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status, 401, "{method} {path} must 401 without a session");
    }
}

/// A malformed brain name is a 400 for every role — validation leaks no
/// existence fact and precedes the role gate (same order as the data plane).
#[tokio::test]
async fn malformed_brain_name_is_400_for_every_role() {
    let app = boot_with_brain().await;
    for cookie in [&app.owner_cookie, &app.viewer_cookie] {
        let (status, body, _) = body_json(
            app.router
                .clone()
                .oneshot(req(
                    "GET",
                    "/_xerj-console/api/v1/graph/BAD_BRAIN/ego?node=x",
                    Some(cookie),
                    None,
                ))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(status, 400, "malformed brain: {body}");
    }
}

/// `ego?nodes_index=` is refused: on the data plane the override is
/// authorized against the caller's grants; a console session holds none, and
/// silently ignoring it would change semantics between the two surfaces.
#[tokio::test]
async fn ego_refuses_the_nodes_index_override() {
    let app = boot_with_brain().await;
    let (status, body, _) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "GET",
                "/_xerj-console/api/v1/graph/casefile/ego?node=file-01&nodes_index=ax-docs",
                Some(&app.owner_cookie),
                None,
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 400, "nodes_index override: {body}");
}

/// The nodes search is pinned to the brain's own meta-doc set: another index
/// on the node — even one that exists — is the same 404 an unknown brain
/// gets, so the endpoint cannot be turned into a general search proxy.
#[tokio::test]
async fn nodes_search_is_pinned_to_the_brains_indices() {
    let app = boot_with_brain().await;
    let (status, _, _) = body_json(
        app.router
            .clone()
            .oneshot(req(
                "POST",
                "/_xerj-console/api/v1/graph/casefile/nodes/_search?index=other-index",
                Some(&app.owner_cookie),
                Some(json!({ "query": { "match_all": {} } })),
            ))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(status, 404, "a foreign index name must 404");
}
