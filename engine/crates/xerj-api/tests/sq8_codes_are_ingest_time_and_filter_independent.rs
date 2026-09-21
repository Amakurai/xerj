//! Issue #392 — `scalar8` codes must be written at INGEST time, addressed by
//! a dense slot, and scored against a codebook that is a property of the
//! index rather than of the query's candidate set.
//!
//! Two observable consequences, each pinned here from the outside over real
//! HTTP:
//!
//!  * `sq8_codes_exist_at_ingest_before_any_query` — after indexing (and
//!    before a single kNN), `GET /{index}/_stats` reports a live, ready SQ8
//!    code store for the field. Before #392 nothing quantized anything at
//!    ingest: the serving path read each candidate's f32 vector out of
//!    `_source` and quantized it per query, so no such store existed and the
//!    stats section is absent (null).
//!
//!  * `scores_do_not_depend_on_the_filter` — the same document must carry the
//!    same `_score` under two queries whose only difference is a filter that
//!    removes OTHER documents. Before #392 the SQ8 codebook was fitted per
//!    query over the candidates being scored, so shrinking the candidate set
//!    moved every surviving document's score (measured on the unfixed code:
//!    max |Δ_score| 1.976e-05 with order swaps at 19/30 positions — inside
//!    SQ8's advertised approximation error, but a real behavioural difference
//!    from Elasticsearch, where the codebook is fixed at index time).
//!
//!  * `codes_survive_a_restart` — the code store is re-derived from the live
//!    documents when the index reopens (WAL replay never re-runs vector
//!    indexing), and once the rebuild lands the same query returns the same
//!    neighbours. A regression pin for the rebuild, not a fail-before: the
//!    per-query-fit path that preceded it also survives a restart.
//!
//!  * `a_wrong_dimension_document_does_not_break_the_field` — a document
//!    whose vector has the wrong dimensionality breaks the store's coverage
//!    invariant; queries must then fall back to the exact path and still
//!    return correct results (the malformed document is skipped, exactly as
//!    the brute-force scan skips it).
//!
//! Elasticsearch is referenced for wire semantics only. It is AGPL-3.0/
//! SSPL-1.0/Elastic-2.0 licensed and no code from it is reproduced here.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

const DIM: usize = 8;
const INDEX: &str = "sq8";

/// 25 corpus documents whose every dimension lies in `[0, 1)` — the documents
/// both queries score. Deterministic prime-stride walk, all values distinct
/// and not representable on both SQ8 grids (that is what makes the unfixed
/// code's score drift observable rather than coincidentally zero).
const KEEP: usize = 25;

/// 5 documents carrying `tag: "wide"` whose dimension 0 sits far outside the
/// keep documents' range. They widen the per-query-fitted codebook only when
/// present, which is exactly the candidate-set dependency under test.
const WIDE_IDS: [usize; 5] = [100, 101, 102, 103, 104];

fn keep_vector(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| ((i * 37 + d * 101) % 199) as f32 / 199.0)
        .collect()
}

fn wide_vector(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[0] = 5.0 + i as f32;
    for d in 1..DIM {
        v[d] = ((i * 53 + d * 97) % 199) as f32 / 199.0;
    }
    v
}

fn doc(i: usize) -> Value {
    if WIDE_IDS.contains(&i) {
        json!({ "tag": "wide", "v": wide_vector(i) })
    } else {
        json!({ "tag": "keep", "v": keep_vector(i) })
    }
}

async fn app() -> (axum::Router, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = xerj_common::config::Config::default();
    config.server.data_dir = dir.path().to_string_lossy().into_owned();
    config.storage.wal_sync = xerj_common::config::WalSync::Async;
    let metrics = xerj_common::metrics::Metrics::new().expect("metrics");
    let engine = xerj_engine::Engine::new(config.clone()).expect("engine");
    let state = xerj_api::state::AppState::new(config, engine, metrics);
    (xerj_api::router::build_es_compat_router(state), dir)
}

async fn send(app: &axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(req).await.expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

async fn json_req(
    app: &axum::Router,
    method: &str,
    path: &str,
    body: Value,
) -> (StatusCode, Value) {
    send(
        app,
        Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request"),
    )
    .await
}

async fn get(app: &axum::Router, path: &str) -> (StatusCode, Value) {
    send(
        app,
        Request::get(path).body(Body::empty()).expect("request"),
    )
    .await
}

async fn create_and_index(app: &axum::Router) {
    let (status, body) = json_req(
        app,
        "PUT",
        &format!("/{INDEX}"),
        json!({
            "mappings": { "properties": {
                "v": { "type": "dense_vector", "dims": DIM, "quantization": "scalar8" }
            } }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "create: {body}");
    for i in 0..KEEP {
        let (status, body) = json_req(app, "PUT", &format!("/{INDEX}/_doc/{i}"), doc(i)).await;
        assert!(status.is_success(), "index {i}: {status} {body}");
    }
    for &i in &WIDE_IDS {
        let (status, body) = json_req(app, "PUT", &format!("/{INDEX}/_doc/{i}"), doc(i)).await;
        assert!(status.is_success(), "index {i}: {status} {body}");
    }
    let (status, body) = json_req(app, "POST", &format!("/{INDEX}/_refresh"), json!({})).await;
    assert!(status.is_success(), "refresh: {status} {body}");
}

/// The query vector: keep document 0's own vector. All keep documents score
/// distinctly against it (deterministic corpus), so score identity between
/// the two queries is not satisfiable by accident.
fn query_vector() -> Vec<f32> {
    keep_vector(0)
}

/// Run the kNN, returning `(id, _score)` per hit.
async fn knn(app: &axum::Router, filter: Option<Value>) -> Vec<(String, f64)> {
    let mut knn_body = json!({
        "field": "v",
        "query_vector": query_vector(),
        "k": KEEP + WIDE_IDS.len(),
        "num_candidates": 100
    });
    if let Some(f) = filter {
        knn_body["filter"] = f;
    }
    let (status, body) = json_req(
        app,
        "POST",
        &format!("/{INDEX}/_search"),
        json!({ "knn": knn_body, "size": KEEP + WIDE_IDS.len(), "_source": false }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "knn: {status} {body}");
    body["hits"]["hits"]
        .as_array()
        .unwrap_or_else(|| panic!("no hits: {body}"))
        .iter()
        .map(|h| {
            (
                h["_id"].as_str().expect("_id").to_string(),
                h["_score"].as_f64().expect("_score"),
            )
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Codes exist at ingest — before any query is served
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn sq8_codes_exist_at_ingest_before_any_query() {
    let (app, _dir) = app().await;
    create_and_index(&app).await;

    // No kNN has run. If SQ8 codes are written at ingest, the per-index stats
    // section reports them; on the per-query-fit path this whole section is
    // absent because nothing exists to report.
    let (status, body) = get(&app, &format!("/{INDEX}/_stats")).await;
    assert_eq!(status, StatusCode::OK, "stats: {status} {body}");
    let field = &body["indices"][INDEX]["primaries"]["sq8"]["fields"]["v"];
    assert!(
        field.is_object(),
        "no ingest-time SQ8 code store in stats (issue #392): {body}"
    );
    assert_eq!(field["dim"], json!(DIM), "{body}");
    assert_eq!(
        field["live"],
        json!(KEEP + WIDE_IDS.len()),
        "every indexed document must hold a live code slot: {body}"
    );
    assert_eq!(
        field["expected"],
        json!(KEEP + WIDE_IDS.len()),
        "coverage numerator and denominator must agree: {body}"
    );
    assert_eq!(field["ready"], json!(true), "{body}");
    // 1 byte per dimension per live document — the resident array itself.
    assert_eq!(
        field["codes_bytes"],
        json!((KEEP + WIDE_IDS.len()) * DIM),
        "{body}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. A document's _score does not depend on the filter
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn scores_do_not_depend_on_the_filter() {
    let (app, _dir) = app().await;
    create_and_index(&app).await;

    let unfiltered = knn(&app, None).await;
    let filtered = knn(&app, Some(json!({ "term": { "tag": "keep" } }))).await;

    assert_eq!(unfiltered.len(), KEEP + WIDE_IDS.len());
    assert_eq!(filtered.len(), KEEP, "filter must remove the wide docs");

    let score_of = |hits: &[(String, f64)], id: &str| {
        hits.iter()
            .find(|(hit_id, _)| hit_id == id)
            .map(|(_, s)| *s)
            .unwrap_or_else(|| panic!("{id} missing"))
    };

    let mut drift: f64 = 0.0;
    for i in 0..KEEP {
        let id = i.to_string();
        let a = score_of(&unfiltered, &id);
        let b = score_of(&filtered, &id);
        drift = drift.max((a - b).abs());
        assert_eq!(
            a, b,
            "doc {id} scores {a} unfiltered but {b} filtered — the SQ8 \
             codebook is being fitted per query over the candidate set, so \
             _score depends on the filter (issue #392)"
        );
    }
    // Belt: on the unfixed path the two fits differ by the wide docs' whole
    // range on dimension 0 (≥ 5.0 over 255 levels ⇒ Δ ≫ 1e-3), so a partial
    // regression still trips the assert above long before this line.
    assert!(drift == 0.0, "drift must be exactly zero, got {drift}");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. The code store is re-derived when the index reopens
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn codes_survive_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = xerj_common::config::Config::default();
    config.server.data_dir = dir.path().to_string_lossy().into_owned();
    config.storage.wal_sync = xerj_common::config::WalSync::Async;

    let before = {
        let metrics = xerj_common::metrics::Metrics::new().expect("metrics");
        let engine = xerj_engine::Engine::new(config.clone()).expect("engine");
        let state = xerj_api::state::AppState::new(config.clone(), engine, metrics);
        let app = xerj_api::router::build_es_compat_router(state.clone());
        create_and_index(&app).await;
        let hits = knn(&app, None).await;
        state.engine.flush_all_force().await;
        hits
    };

    let (app, _keep) = {
        let metrics = xerj_common::metrics::Metrics::new().expect("metrics");
        let engine = xerj_engine::Engine::new(config.clone()).expect("engine");
        let state = xerj_api::state::AppState::new(config.clone(), engine, metrics);
        let app = xerj_api::router::build_es_compat_router(state.clone());
        // The open-time rebuild is background work; wait for it so the
        // assertion observes the codes path, not a mid-rebuild fallback.
        if let Ok(idx) = state.engine.get_index(INDEX) {
            idx.await_sq8_rebuilds().await;
        }
        (app, ())
    };

    let after = knn(&app, None).await;

    // Same neighbours in the same order: quantizing the same live vectors
    // under a codebook spanning the same per-dimension ranges reproduces the
    // ranking. Scores may shift by at most the SQ8 quantization step (the
    // re-fit walk re-encodes from the true f32 sources, while the live path
    // may have re-encoded through a decode), so ids/order are asserted
    // strictly and scores within a generous step-sized tolerance.
    assert_eq!(
        before.len(),
        after.len(),
        "restart must not lose documents: {before:?} vs {after:?}"
    );
    let order_before: Vec<&str> = before.iter().map(|(id, _)| id.as_str()).collect();
    let order_after: Vec<&str> = after.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(
        order_before, order_after,
        "ranking changed across restart: {before:?} vs {after:?}"
    );
    for ((_, a), (_, b)) in before.iter().zip(after.iter()) {
        assert!(
            (a - b).abs() < 1e-2,
            "score for the same document moved from {a} to {b} across restart"
        );
    }

    // And the re-derived store is again serving-shape.
    let (status, body) = get(&app, &format!("/{INDEX}/_stats")).await;
    assert_eq!(status, StatusCode::OK);
    let field = &body["indices"][INDEX]["primaries"]["sq8"]["fields"]["v"];
    assert_eq!(field["ready"], json!(true), "{body}");
    assert_eq!(field["live"], json!(KEEP + WIDE_IDS.len()), "{body}");
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Coverage broken → exact fallback, correct results
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_wrong_dimension_document_does_not_break_the_field() {
    let (app, _dir) = app().await;
    create_and_index(&app).await;

    // A document whose vector has the wrong dimensionality. The brute-force
    // scan has always skipped it (`doc_vec.len() != dim`); the code store
    // must not serve around it either — coverage breaks and queries fall
    // back to the exact path rather than returning wrong results.
    let (status, body) = json_req(
        &app,
        "PUT",
        &format!("/{INDEX}/_doc/broken"),
        json!({ "tag": "keep", "v": vec![0.5f32; DIM + 3] }),
    )
    .await;
    assert!(status.is_success(), "index broken: {status} {body}");
    let (status, body) = json_req(&app, "POST", &format!("/{INDEX}/_refresh"), json!({})).await;
    assert!(status.is_success(), "{status} {body}");

    let hits = knn(&app, Some(json!({ "term": { "tag": "keep" } }))).await;
    assert_eq!(
        hits.len(),
        KEEP,
        "the malformed doc must be skipped: {hits:?}"
    );
    assert!(
        !hits.iter().any(|(id, _)| id == "broken"),
        "malformed document must not appear in hits: {hits:?}"
    );
    // The control: doc 0 is the query vector and must still top the list.
    assert_eq!(hits[0].0, "0", "query-vector doc must rank first: {hits:?}");
    assert!(hits[0].1 > 0.99, "expected ~1.0, got {}", hits[0].1);
}
