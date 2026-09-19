//! Regression test for #940 — a `hybrid` page must be a function of the index,
//! not of the process that served it.
//!
//! Tied fused scores are structural under RRF, not an edge case: a document
//! found only by leg A at rank r and one found only by leg B at rank r both
//! score exactly `w/(k+r)`. `fuse_rrf`/`fuse_linear` used to accumulate into a
//! `HashMap` and drain it into a score-only stable sort, so every such tie
//! came out in hash-iteration order, and `std`'s `RandomState` is seeded per
//! process. Measured on BEIR SciFact (issue #940): after a restart on
//! unchanged data 32 of 40 queries changed order and 21 of 40 changed their
//! top 10, with identical hit sets.
//!
//! The order asserted here is the ONE total order every score-ranked page
//! uses (#270): `score DESC, seq_no ASC (arrival), _id ASC`.
//!
//! Two things this test is careful about:
//!
//!   * **The result cache.** `Index::query_cache` is keyed by
//!     `(query hash, dataset version)`, so a repeated request is served from
//!     it without re-running fusion — which is why the issue's own in-process
//!     repeats looked stable. It is cleared before every search below; a test
//!     that did not would pass against the broken code.
//!   * **`_id` order is not arrival order.** Documents are written in a
//!     scrambled order, so a fix that tied on `_id` alone — or any order that
//!     merely happened to be repeatable — fails the arrival assertion.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use tempfile::TempDir;
use xerj_common::config::Config;
use xerj_common::types::Schema;
use xerj_engine::index::Index;
use xerj_engine::Engine;
use xerj_query::parse_request;

const INDEX: &str = "hy940";
/// Documents per leg. Each rank r in 0..PER_LEG yields one exact tie pair.
const PER_LEG: usize = 12;
/// Tokens per document — constant, so BM25's length normalisation is the
/// same for every document and the score is a function of `tf` alone.
const DOC_LEN: usize = 16;

fn make_engine(dir: &TempDir) -> Engine {
    let mut config = Config::default();
    config.server.data_dir = dir.path().to_str().unwrap().to_string();
    Engine::new(config).expect("engine::new")
}

fn hybrid(fusion: Value) -> Value {
    json!({
        "hybrid": {
            "queries": [
                { "query": { "match": { "body": "alpha" } }, "weight": 1.0 },
                { "query": { "match": { "body": "beta"  } }, "weight": 1.0 }
            ],
            "fusion": fusion
        }
    })
}

fn rrf() -> Value {
    hybrid(json!({ "type": "rrf", "k": 60 }))
}

fn linear() -> Value {
    hybrid(json!("linear"))
}

/// `word` repeated `tf` times, padded with a filler token to `DOC_LEN`.
fn body(word: &str, tf: usize) -> String {
    let mut tokens = vec![word; tf];
    tokens.resize(DOC_LEN, "filler");
    tokens.join(" ")
}

/// A symmetric two-leg corpus: `a-<r>` holds "alpha" `PER_LEG - r` times and
/// `b-<r>` holds "beta" the same number of times, so inside each leg the
/// scores are DISTINCT and strictly decreasing in r (the leg's rank order
/// owes nothing to any tie-break), while across legs `a-<r>` and `b-<r>`
/// sit at the same rank and tie exactly after fusion.
///
/// Written in a scrambled order. Returns id → arrival position.
async fn symmetric_corpus(idx: &Arc<Index>) -> HashMap<String, usize> {
    let mut ids: Vec<(String, String)> = Vec::new();
    for r in 0..PER_LEG {
        ids.push((format!("a-{r:02}"), body("alpha", PER_LEG - r)));
        ids.push((format!("b-{r:02}"), body("beta", PER_LEG - r)));
    }
    // A fixed permutation (multiplication by 7 is a bijection mod 24), so
    // arrival order agrees with neither `_id` order nor rank order — and
    // within a tie pair it is `a` first for some ranks and `b` first for
    // others.
    let n = ids.len();
    let mut arrival = HashMap::new();
    for pos in 0..n {
        let (id, text) = ids[(pos * 7 + 5) % n].clone();
        idx.index_document(Some(id.clone()), json!({ "body": text }))
            .await
            .unwrap();
        arrival.insert(id, pos);
    }
    assert_eq!(
        arrival.len(),
        n,
        "the permutation must visit every document"
    );
    arrival
}

/// One uncached hybrid search → `(id, score bits)` per hit.
async fn page(idx: &Arc<Index>, from: usize, size: usize, q: &Value) -> Vec<(String, u32)> {
    idx.query_cache.clear();
    let req = parse_request(&json!({ "from": from, "size": size, "query": q })).unwrap();
    let res = idx.search(&req).await.unwrap();
    res.hits
        .into_iter()
        .map(|h| (h.id, h.score.to_bits()))
        .collect()
}

/// Inside every run of bit-identical scores, hits must be in arrival order.
fn assert_ties_are_in_arrival_order(
    label: &str,
    hits: &[(String, u32)],
    arrival: &HashMap<String, usize>,
) {
    for w in hits.windows(2) {
        if w[0].1 == w[1].1 {
            assert!(
                arrival[&w[0].0] < arrival[&w[1].0],
                "{label}: {} and {} tie on score but are not in arrival order \
                 (arrived {} and {}) — full page: {hits:?}",
                w[0].0,
                w[1].0,
                arrival[&w[0].0],
                arrival[&w[1].0],
            );
        }
    }
}

/// The page RRF must return for `symmetric_corpus`: rank by rank, and inside
/// each rank's tie pair the document that arrived first.
fn expected_rrf_order(arrival: &HashMap<String, usize>) -> Vec<String> {
    let mut out = Vec::new();
    for r in 0..PER_LEG {
        let mut pair = [format!("a-{r:02}"), format!("b-{r:02}")];
        pair.sort_by_key(|id| arrival[id]);
        out.extend(pair);
    }
    out
}

fn ids_of(hits: &[(String, u32)]) -> Vec<String> {
    hits.iter().map(|(id, _)| id.clone()).collect()
}

/// Everything a paginating client relies on, checked against one open index.
async fn assert_fused_pages(label: &str, idx: &Arc<Index>, arrival: &HashMap<String, usize>) {
    let n = 2 * PER_LEG;
    let expected = expected_rrf_order(arrival);

    // ── RRF ──────────────────────────────────────────────────────────────
    let full = page(idx, 0, n, &rrf()).await;
    assert_eq!(full.len(), n, "{label}: both legs fully retrieved");
    // The fixture really does tie at every rank — otherwise this test would
    // pass against the broken code and prove nothing.
    for (r, pair) in full.chunks(2).enumerate() {
        assert_eq!(
            pair[0].1, pair[1].1,
            "{label}: rank {r} must be an EXACT fused tie, got {pair:?}"
        );
    }
    assert_eq!(
        ids_of(&full),
        expected,
        "{label}: tied fused scores must resolve by arrival order"
    );

    // Re-running fusion must reproduce the page. 32 uncached repeats: under
    // the old per-map `RandomState` the 12 tie pairs agreed between two runs
    // with probability 2^-12.
    for run in 0..32 {
        assert_eq!(
            page(idx, 0, n, &rrf()).await,
            full,
            "{label}: uncached repeat {run} returned a different page"
        );
    }

    // Every bounded page is a prefix of the full page …
    for size in 1..=n {
        assert_eq!(
            page(idx, 0, size, &rrf()).await,
            full[..size],
            "{label}: size:{size} disagrees with the full page"
        );
    }
    // … and walking it with `from` reproduces it, including the pages whose
    // boundary falls INSIDE a tie pair (every odd `from`). `from + size`
    // stays under the 50-deep leg window, so every page fuses the same legs.
    for size in [1usize, 3, 5] {
        let mut walked = Vec::new();
        let mut from = 0;
        while from < n {
            walked.extend(page(idx, from, size, &rrf()).await);
            from += size;
        }
        assert_eq!(
            walked, full,
            "{label}: paging by {size} reorders the corpus"
        );
    }

    // ── Linear ───────────────────────────────────────────────────────────
    // Same accumulator, same defect. The legs are symmetric, so min-max
    // normalisation maps `a-<r>` and `b-<r>` to the same value and they tie
    // here too; the assertion is the general one so it does not depend on
    // that.
    let lin = page(idx, 0, n, &linear()).await;
    assert_eq!(lin.len(), n, "{label}: linear retrieves both legs");
    assert_ties_are_in_arrival_order(&format!("{label}/linear"), &lin, arrival);
    assert!(
        lin.windows(2).any(|w| w[0].1 == w[1].1),
        "{label}: the linear fixture is expected to tie somewhere, got {lin:?}"
    );
    for run in 0..32 {
        assert_eq!(
            page(idx, 0, n, &linear()).await,
            lin,
            "{label}: linear uncached repeat {run} returned a different page"
        );
    }
}

/// Flushed data: the page is the same before the restart, after it, and after
/// the one after that.
#[tokio::test]
async fn hybrid_page_survives_restarts_on_flushed_data() {
    let dir = TempDir::new().unwrap();
    let arrival;
    let before;
    {
        let engine = make_engine(&dir);
        engine.create_index(INDEX, Schema::empty()).unwrap();
        let idx = engine.get_index(INDEX).unwrap();
        arrival = symmetric_corpus(&idx).await;
        idx.flush().await.unwrap();
        assert_fused_pages("flushed", &idx, &arrival).await;
        before = (
            page(&idx, 0, 2 * PER_LEG, &rrf()).await,
            page(&idx, 0, 2 * PER_LEG, &linear()).await,
        );
        drop(idx);
        drop(engine);
    }
    for restart in 1..=3 {
        let engine = make_engine(&dir);
        let idx = engine.get_index(INDEX).unwrap();
        let label = format!("flushed/restart-{restart}");
        assert_fused_pages(&label, &idx, &arrival).await;
        let after = (
            page(&idx, 0, 2 * PER_LEG, &rrf()).await,
            page(&idx, 0, 2 * PER_LEG, &linear()).await,
        );
        assert_eq!(after, before, "{label}: page changed across a restart");
        drop(idx);
        drop(engine);
    }
}

/// Unflushed data: after the restart the documents come back through WAL
/// replay into the memtable, a different read path from the segments above.
#[tokio::test]
async fn hybrid_page_survives_restarts_on_unflushed_data() {
    let dir = TempDir::new().unwrap();
    let arrival;
    let before;
    {
        let engine = make_engine(&dir);
        engine.create_index(INDEX, Schema::empty()).unwrap();
        let idx = engine.get_index(INDEX).unwrap();
        arrival = symmetric_corpus(&idx).await;
        assert_fused_pages("memtable", &idx, &arrival).await;
        before = page(&idx, 0, 2 * PER_LEG, &rrf()).await;
        drop(idx);
        drop(engine);
    }
    for restart in 1..=2 {
        let engine = make_engine(&dir);
        let idx = engine.get_index(INDEX).unwrap();
        let label = format!("memtable/restart-{restart}");
        assert_fused_pages(&label, &idx, &arrival).await;
        assert_eq!(
            page(&idx, 0, 2 * PER_LEG, &rrf()).await,
            before,
            "{label}: page changed across a restart"
        );
        drop(idx);
        drop(engine);
    }
}

/// A flush in the middle moves half the corpus to segments. Arrival order is
/// a property of the documents, not of where they are stored, so the page
/// must not notice.
#[tokio::test]
async fn hybrid_page_does_not_depend_on_where_the_documents_are_stored() {
    let dir = TempDir::new().unwrap();
    let engine = make_engine(&dir);
    engine.create_index(INDEX, Schema::empty()).unwrap();
    let idx = engine.get_index(INDEX).unwrap();
    let arrival = symmetric_corpus(&idx).await;

    let in_memtable = page(&idx, 0, 2 * PER_LEG, &rrf()).await;
    idx.flush().await.unwrap();
    let in_segments = page(&idx, 0, 2 * PER_LEG, &rrf()).await;

    assert_eq!(ids_of(&in_memtable), expected_rrf_order(&arrival));
    assert_eq!(
        ids_of(&in_segments),
        ids_of(&in_memtable),
        "a flush reordered a hybrid page"
    );
}
