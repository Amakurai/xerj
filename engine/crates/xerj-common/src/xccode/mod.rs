//! # xccode — the reference-coding semantics shared by `xerj code`, the
//! `xerj corpus` lifecycle, and the `xerj_code_search` MCP tool
//!
//! Until issue #977 the reference-coding loop lived in three wrapper scripts
//! (`tools/xerj-code/scripts/xc.py`, `xc-index.sh`, `xc-corpus.sh`). They are
//! ported here as ONE pure implementation so the CLI and the MCP tool serve
//! byte-identical prose from the same renderer, and so every contract — the
//! 30-day staleness refusal, the restricted-licence warning, the
//! not-loaded-here diagnosis, the exit-code triangle — has exactly one home
//! and one test suite.
//!
//! Layout (all persistent paths parameterised by `root`; tests inject
//! tempdirs — that is test hygiene, NOT a layout change):
//!
//! * [`state`] — the `state/<corpus>.json` ledger: read, staleness,
//!   incomplete-coverage warning, atomic write
//! * [`licence`] — restricted-licence tuple + the clone-time detector
//! * [`manifest`] — `corpora/<corpus>/corpus.json` read/write + hub manifests
//! * [`pathgate`] — the hard gate every manifest-derived path passes before
//!   anything destructive runs inside it
//! * [`fields`] — mapping-resolved `multi_match` fields + semantic-capable
//!   index discovery
//! * [`passage`] — symbol-passage / best-window / file-head selection
//! * [`render`] — the one output renderer (prose + MEATL)
//!
//! HTTP is a 3-method shell ([`XcHttp`]) so the pure core stays free of any
//! client dependency: `xerj-autoindex` implements it over its blocking `Es`
//! client, `xerj-mcp` over reqwest. No fusion or query logic lives in either
//! — hybrid retrieval emits the engine's native top-level `hybrid` query and
//! lets the server fuse.

pub mod fields;
pub mod licence;
pub mod manifest;
pub mod passage;
pub mod pathgate;
pub mod render;
pub mod state;

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

/// Corpora older than this are refused, not served: a stale index returns
/// code that no longer exists, with false confidence (`STALE_DAYS` in the
/// original `xc.py`).
pub const STALE_DAYS: i64 = 30;

/// Reciprocal Rank Fusion constant (Cormack, Clarke & Buettcher, SIGIR 2009).
/// k=60 is the paper's value and the universal default. It is NOT tuned here.
/// Fusion itself is SERVER-SIDE (native `hybrid` query, #943); this constant
/// only names k in the request we emit and in the arms-ran note.
pub const RRF_K: u64 = 60;

/// The retrieval arm. Default is [`Mode::Bm25`] — measured 12/12 top-3 across
/// both standing corpora (bm25 beat hybrid and semantic on the combined
/// metric; see `tools/xerj-code/SKILL.md`'s retrieval-mode table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Bm25,
    Semantic,
    Hybrid,
}

impl Mode {
    /// Parse the `--mode` / MCP `mode` argument. Unknown values are usage
    /// errors (exit 2), never a silent default.
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "bm25" => Some(Mode::Bm25),
            "semantic" => Some(Mode::Semantic),
            "hybrid" => Some(Mode::Hybrid),
            _ => None,
        }
    }
}

/// Everything one `xerj code` invocation (or `xerj_code_search` tool call)
/// needs. Defaults mirror `xc.py` exactly: k=5, bm25, full=800, symbols on.
#[derive(Debug, Clone)]
pub struct CodeParams {
    pub corpus: String,
    pub query: String,
    pub k: usize,
    pub lang: Option<String>,
    pub mode: Mode,
    /// Max chars of each passage; 0 = file-head mode (first 400 chars,
    /// labelled as the file head — retrieving the definition IS the feature,
    /// issue #368, so this is a cap, never an opt-out).
    pub full: usize,
    pub no_symbol: bool,
    pub stale_ok: bool,
    pub meatl: bool,
    /// CLI `--json`: raw server response on stdout. MCP never sets this.
    pub as_json: bool,
    /// MCP `licence_policy:"strict"`: restricted-licence hits keep
    /// provenance + licence + warning but lose the passage text. The CLI
    /// stays `false` (warn) for `xc.py` parity.
    pub strict_licence: bool,
}

impl CodeParams {
    pub fn new(corpus: &str, query: &str) -> Self {
        CodeParams {
            corpus: corpus.to_string(),
            query: query.to_string(),
            k: 5,
            lang: None,
            mode: Mode::Bm25,
            full: 800,
            no_symbol: false,
            stale_ok: false,
            meatl: false,
            as_json: false,
            strict_licence: false,
        }
    }
}

/// The one HTTP shell the pure core needs. Implemented over the blocking `Es`
/// client in `xerj-autoindex` and over reqwest in `xerj-mcp`; fakes in tests.
///
/// `cat_indices_json` maps HTTP 404 to `Ok(vec![])` — a wildcard that matches
/// no index means "none", not an error — while every other failure is `Err`:
/// a node that cannot answer must never be reported as "0 live indices".
pub trait XcHttp {
    /// `GET /{pattern}/_mapping`. `Err` carries a human-readable reason.
    fn get_mapping(&self, pattern: &str) -> Result<Value, String>;
    /// `GET /_cat/indices/{pattern}?format=json&h=index` as index names.
    /// `Err` = unreachable/ambiguous (never a silent zero).
    fn cat_indices_json(&self, pattern: &str) -> Result<Vec<String>, String>;
    /// `POST /{index}/_search`. `Err` carries status + reason or transport.
    fn search(&self, index: &str, body: &Value) -> Result<Value, String>;
}

/// Everything a caller needs after the pipeline ran. The CLI prints `text`
/// (stdout, or stderr when [`Self::to_stderr`] — refusals are diagnostics)
/// and exits with [`Self::exit`]; MCP returns `text` as the tool payload with
/// [`Self::is_error`]. Warnings ride stderr on the CLI and the TOP of the
/// tool text on MCP — MCP has no stderr channel, so warnings are content.
#[derive(Debug, Clone)]
pub struct CodeOutcome {
    pub text: String,
    pub warnings: Vec<String>,
    /// 0 = hits, 1 = no match (a miss is an answer), 2 = usage / stale /
    /// transport, 3 = corpus in state/ but 0 live indices on this node.
    pub exit: i32,
    /// MCP `isError`: true exactly for the action-needed states (2 and 3).
    /// A no-match is `false` with the fall-back prose.
    pub is_error: bool,
    /// Raw server response when `as_json` was requested (CLI-only flag).
    pub json: Option<Value>,
    pub to_stderr: bool,
}

impl CodeOutcome {
    fn refused(text: String, exit: i32) -> Self {
        CodeOutcome {
            text,
            warnings: Vec::new(),
            exit,
            is_error: true,
            json: None,
            to_stderr: true,
        }
    }
}

/// The whole query pipeline, pure: ledger → staleness → live-count → mapping
/// discovery → query → passage selection → licence map → render.
///
/// `url` is the resolved node URL (for diagnostics); `stale_hint` is the
/// override spelling the caller's surface accepts — "`--stale-ok`" on the
/// CLI, "`stale_ok:true`" on MCP — so the one refusal text names the fix the
/// caller can actually type.
pub fn run_code_query(
    root: &Path,
    http: &impl XcHttp,
    url: &str,
    params: &CodeParams,
    stale_hint: &str,
) -> CodeOutcome {
    let corpus = params.corpus.clone();
    let query = params.query.clone();

    // 1. The ledger. Not indexed is a usage-class refusal naming the command
    //    that fixes it (exit 2 — the caller's driver script should stop).
    let st = match state::load_state(root, &corpus) {
        Ok(st) => st,
        Err(msg) => return CodeOutcome::refused(msg, 2),
    };

    // 2. Staleness: refuse BEFORE any query. A stale index returns code that
    //    no longer exists; --stale-ok / stale_ok:true is the only override
    //    (no env bypass — a wrapper must not be able to de-fang this).
    let age_days = match state::check_fresh(&st, params.stale_ok, stale_hint) {
        Ok(age) => age,
        Err(refusal) => return CodeOutcome::refused(refusal, 2),
    };

    let prefix = state::query_prefix(&st);

    // 3. Not-loaded-here: in state/ yet 0 live indices on THIS node. A
    //    distinct, actionable diagnosis (exit 3) — collapsing it into
    //    "no match" is the state-ledger trust trap this guard closes.
    //    An unreachable node is reported honestly (exit 2) by the search
    //    path below, never as "0 live indices".
    match http.cat_indices_json(&format!("{prefix}*")) {
        Ok(indices) if indices.is_empty() => {
            return CodeOutcome::refused(state::not_loaded_message(&st, &prefix, url), 3)
        }
        // Unreachable/ambiguous: fall through; the search itself will report
        // the transport failure honestly.
        _ => {}
    }

    let mut warnings = Vec::new();
    if let Some(w) = state::incomplete_coverage(&st) {
        warnings.push(w);
    }

    // 4. Mapping: one read serves both field resolution and capable-set
    //    discovery. Unreadable is data, not an error (bm25 degrades to the
    //    full field list; hybrid degrades to BM25-only).
    let pattern = format!("{prefix}*");
    let mapping = http.get_mapping(&format!("/{pattern}/_mapping")).ok();
    let fields = fields::resolve_fields(mapping.as_ref());

    let mut note: Option<String> = None;
    let mut rrf_scores = false;

    let (resp, hits) = match params.mode {
        Mode::Bm25 => match http.search(
            &pattern,
            &bm25_body(&query, params.k, &params.lang, &fields),
        ) {
            Ok(r) => (r.clone(), hit_list(&r)),
            Err(e) => return CodeOutcome::refused(format!("search failed: {e}"), 2),
        },
        Mode::Semantic => {
            // Standalone semantic is fatal on mapping/search failure: it has
            // no BM25 result to keep.
            let mapping = match http.get_mapping(&format!("/{pattern}/_mapping")) {
                Ok(m) => m,
                Err(e) => {
                    return CodeOutcome::refused(
                        format!("semantic mapping lookup failed at {url}: {e}"),
                        2,
                    )
                }
            };
            let (capable, total) = fields::semantic_capable(Some(&mapping));
            if capable.is_empty() {
                // No semantic arm is possible: do NOT post the query. An
                // empty comma-join is an empty index segment, which the
                // server reads as `/_search` — every index on the node,
                // corpus or not. The empty answer with its note is the
                // honest result.
                note = Some(format!(
                    "no index under '{pattern}' maps `body` as semantic_text"
                ));
                (Value::Null, Vec::new())
            } else {
                note = Some(format!(
                    "vector only over {} of {total} index(es)",
                    capable.len()
                ));
                let body = semantic_body(&query, params.k, &params.lang);
                match http.search(&capable.join(","), &body) {
                    Ok(r) => (r.clone(), hit_list(&r)),
                    Err(e) => return CodeOutcome::refused(format!("search failed: {e}"), 2),
                }
            }
        }
        Mode::Hybrid => {
            let (capable, total) = fields::semantic_capable(mapping.as_ref());
            if capable.is_empty() {
                // No vector arm possible: degrade, and SAY so.
                note = Some(format!(
                    "BM25 only — no usable semantic_text mapping for `body` could be \
                     discovered under '{pattern}'"
                ));
                match http.search(
                    &pattern,
                    &bm25_body(&query, params.k, &params.lang, &fields),
                ) {
                    Ok(r) => (r.clone(), hit_list(&r)),
                    Err(e) => return CodeOutcome::refused(format!("search failed: {e}"), 2),
                }
            } else {
                // BM25 preflight: the lexical embedder returns confident
                // neighbours for ANY input, so a BM25 miss is the corpus's
                // only honest "no" — vector hits must not launder it.
                match http.search(&pattern, &bm25_body(&query, 1, &params.lang, &fields)) {
                    Ok(pre) if hit_list(&pre).is_empty() => {
                        note = Some(
                            "no lexical match in this corpus — vector nearest-neighbours \
                             are not evidence of a match, so this is reported as a miss"
                                .to_string(),
                        );
                        (Value::Null, Vec::new())
                    }
                    Ok(_) => {
                        // ONE fused request: the engine's native top-level
                        // `hybrid` (RRF k=60) — no client-side fusion. Aimed
                        // at the comma-joined CAPABLE set when only some
                        // indices map `body` as semantic_text: a semantic leg
                        // posted at a wildcard covering plain-text indices
                        // 400s the WHOLE request.
                        let target = if capable.len() == total {
                            pattern.clone()
                        } else {
                            capable.join(",")
                        };
                        let body = hybrid_body(&query, params.k, &params.lang, &fields);
                        match http.search(&target, &body) {
                            Ok(r) => {
                                rrf_scores = true;
                                let mut n = format!(
                                    "hybrid RRF(k={RRF_K}) — BM25 over {total} index(es), \
                                     vector over {} of {total}",
                                    capable.len()
                                );
                                if capable.len() < total {
                                    let excluded: Vec<String> = mapping
                                        .as_ref()
                                        .and_then(|m| {
                                            m.as_object().map(|obj| {
                                                obj.keys()
                                                    .filter(|k| !capable.contains(&k.to_string()))
                                                    .cloned()
                                                    .collect::<Vec<_>>()
                                            })
                                        })
                                        .unwrap_or_default();
                                    if !excluded.is_empty() {
                                        n.push_str(&format!(
                                            "; lexical-only (excluded from the vector arm): \
                                             {}",
                                            excluded.join(", ")
                                        ));
                                    }
                                }
                                note = Some(n);
                                (r.clone(), hit_list(&r))
                            }
                            Err(_) => {
                                // The vector arm failed, not the corpus:
                                // degrade to plain BM25, never abort.
                                note = Some(format!(
                                    "BM25 only — vector search failed or returned no hits \
                                     from the {} semantic_text index(es)",
                                    capable.len()
                                ));
                                match http.search(
                                    &pattern,
                                    &bm25_body(&query, params.k, &params.lang, &fields),
                                ) {
                                    Ok(r) => (r.clone(), hit_list(&r)),
                                    Err(e) => {
                                        return CodeOutcome::refused(
                                            format!("search failed: {e}"),
                                            2,
                                        )
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => return CodeOutcome::refused(format!("search failed: {e}"), 2),
                }
            }
        }
    };

    // 5. --json: the raw response stays machine-readable while the exit-code
    //    contract is unchanged (empty hits STILL exit 1).
    if params.as_json {
        return CodeOutcome {
            text: String::new(),
            warnings,
            exit: if hits.is_empty() { 1 } else { 0 },
            is_error: false,
            json: Some(resp),
            to_stderr: false,
        };
    }

    if hits.is_empty() {
        // Say so explicitly. A silent miss makes the next agent re-run the
        // same dead query; this is the line that stops the loop.
        return CodeOutcome {
            text: render::no_match_text(&corpus, &query, params.meatl),
            warnings,
            exit: 1,
            is_error: false,
            json: None,
            to_stderr: false,
        };
    }

    // 6. Licence map: ALWAYS from the corpus's own corpus.json — never
    //    re-derived at query time, never from prose.
    let licences = manifest::licence_map(root, &corpus);

    let text = render::render(
        &hits,
        &licences,
        &render::RenderOpts {
            corpus: &corpus,
            query: &query,
            full: params.full,
            no_symbol: params.no_symbol,
            meatl: params.meatl,
            rrf_scores,
            age_days,
            strict_licence: params.strict_licence,
        },
        note.as_deref(),
    );

    CodeOutcome {
        text,
        warnings,
        exit: 0,
        is_error: false,
        json: None,
        to_stderr: false,
    }
}

/// Hits of a search response, missing containers tolerated as empty.
pub(crate) fn hit_list(resp: &Value) -> Vec<Value> {
    resp.pointer("/hits/hits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// The measured BM25 body: flat `multi_match` over mapping-resolved fields
/// (`FIELDS`'s flat weights were measured, not chosen — 12/12 top-3). NEVER
/// a highlight block (issue #177: highlight changes ranking on this engine),
/// and no `_source` projection on the corpus path — passage selection needs
/// `body` and `symbols`.
pub(crate) fn bm25_body(query: &str, k: usize, lang: &Option<String>, fields: &[String]) -> Value {
    let mut must = vec![serde_json::json!({
        "multi_match": { "query": query, "fields": fields }
    })];
    if let Some(lg) = lang {
        must.push(serde_json::json!({ "match": { "language": lg } }));
    }
    serde_json::json!({ "size": k, "query": { "bool": { "must": must } } })
}

/// The vector-only body. Aimed ONLY at indices whose `body` is semantic_text
/// (a semantic query against plain text 400s the whole wildcard).
pub(crate) fn semantic_body(query: &str, k: usize, lang: &Option<String>) -> Value {
    let q = serde_json::json!({ "semantic": { "field": "body", "query": query } });
    let query = match lang {
        Some(lg) => serde_json::json!({
            "bool": { "must": [q, { "match": { "language": lg } }] }
        }),
        None => q,
    };
    serde_json::json!({ "size": k, "query": query })
}

/// ONE native top-level `hybrid` request: BM25 clause + semantic clause,
/// fused server-side by RRF(k=60). Hybrid anywhere else is a 400 (#943), so
/// this shape is the whole contract — and there is deliberately NO
/// client-side fusion left to drift from the engine's.
pub(crate) fn hybrid_body(
    query: &str,
    k: usize,
    lang: &Option<String>,
    fields: &[String],
) -> Value {
    // `--lang` must constrain BOTH legs (xc.py semantics): a language filter
    // on BM25 only lets the vector leg surface docs the user filtered out.
    let wrap = |q: Value| -> Value {
        match lang {
            Some(lg) => serde_json::json!({
                "bool": { "must": [q, { "match": { "language": lg } }] }
            }),
            None => q,
        }
    };
    let bm = wrap(serde_json::json!({
        "multi_match": { "query": query, "fields": fields }
    }));
    let sem = wrap(serde_json::json!({
        "semantic": { "field": "body", "query": query }
    }));
    serde_json::json!({
        "size": k,
        "query": { "hybrid": {
            "queries": [
                { "query": bm },
                { "query": sem }
            ],
            "fusion": { "method": "rrf", "k": RRF_K }
        }}
    })
}

/// `xerj corpus list` needs per-corpus live counts without the full pipeline;
/// this is the honest tri-state the ledger listing shares with `require_loaded`.
pub fn live_count_summary(http: &impl XcHttp, prefix: &str) -> Result<usize, String> {
    http.cat_indices_json(&format!("{prefix}*"))
        .map(|v| v.len())
}

/// Licence map for a corpus, exposed for `xerj corpus list`'s review.use line.
pub fn corpus_review_uses(root: &Path, corpus: &str) -> HashMap<String, String> {
    manifest::read_corpus_manifest(&root.join("corpora").join(corpus).join("corpus.json"))
        .map(|m| {
            m.repos
                .iter()
                .filter_map(|r| {
                    r.review
                        .as_ref()
                        .and_then(|v| v.get("use"))
                        .and_then(Value::as_str)
                        .map(|u| (r.repo.clone(), u.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake node: records every request, answers from canned data.
    struct FakeHttp {
        mapping: Value,
        hits: Vec<Value>,
        // (index, body) per search, in order — pins the EXACT wire shape.
        searches: std::sync::Mutex<Vec<(String, Value)>>,
        live: Vec<String>,
        fail_search: bool,
    }

    impl FakeHttp {
        fn new() -> Self {
            FakeHttp {
                mapping: serde_json::json!({
                    "xc-kv-b1-000": { "mappings": { "properties": {
                        "body": { "type": "text" }, "defs": { "type": "text" },
                        "title": { "type": "text" }
                    }}}
                }),
                hits: vec![serde_json::json!({
                    "_score": 4.2,
                    "_source": {
                        "ax_path": "valkey/src/networking.c", "line": 1460,
                        "body": "void addReplyNull(client *c) {\n    addReplyProto(c, \"$-1\\r\\n\", 5);\n}\n",
                        "symbols": [ { "name": "addReplyNull", "kind": "function", "line": 1 } ]
                    }
                })],
                searches: std::sync::Mutex::new(Vec::new()),
                live: vec!["xc-kv-b1-000".to_string()],
                fail_search: false,
            }
        }
    }

    impl XcHttp for FakeHttp {
        fn get_mapping(&self, pattern: &str) -> Result<Value, String> {
            // The real adapter hands `path` to `Es::get_json`, which sends it
            // verbatim after `base` — a path without the leading `/` is a
            // malformed URL, not a 404. Assert the shape here so a regression
            // in the caller fails the tests instead of silently passing.
            assert!(
                pattern.starts_with('/'),
                "get_mapping path must be request-shaped, got {pattern:?}"
            );
            Ok(self.mapping.clone())
        }
        fn cat_indices_json(&self, _pattern: &str) -> Result<Vec<String>, String> {
            Ok(self.live.clone())
        }
        fn search(&self, index: &str, body: &Value) -> Result<Value, String> {
            self.searches
                .lock()
                .unwrap()
                .push((index.to_string(), body.clone()));
            if self.fail_search {
                return Err("transport: connection refused".to_string());
            }
            Ok(serde_json::json!({ "hits": { "hits": self.hits } }))
        }
    }

    fn root_with_state() -> std::path::PathBuf {
        let root = tempfile::tempdir().unwrap().keep();
        std::fs::create_dir_all(root.join("state")).unwrap();
        std::fs::write(
            root.join("state/kv.json"),
            format!(
                "{{\"corpus\":\"kv\",\"indexed_at\":\"{}\",\"prefix\":\"xc-kv\",\"url\":\"u\",\
                 \"autoindex_exit\":0,\"salvaged\":false,\"build\":\"b1\",\
                 \"index_prefix\":\"xc-kv-b1\",\"state_dir\":\"/tmp/s\"}}",
                chrono::Utc::now().format(state::STAMP_FORMAT)
            ),
        )
        .unwrap();
        std::fs::create_dir_all(root.join("corpora/kv/valkey")).unwrap();
        std::fs::write(
            root.join("corpora/kv/corpus.json"),
            "{\"corpus\":\"kv\",\"cloned_at\":\"t\",\"repos\":[{\"repo\":\"valkey\",\"url\":\"u\",\
             \"licence\":\"BSD-3-Clause\",\"sha\":\"s\"}]}",
        )
        .unwrap();
        root
    }

    #[test]
    fn bm25_query_uses_the_star_direct_glob_and_never_highlights() {
        let root = root_with_state();
        let http = FakeHttp::new();
        let out = run_code_query(
            &root,
            &http,
            "http://localhost:9200",
            &CodeParams::new("kv", "addReplyNull"),
            "`--stale-ok`",
        );
        assert_eq!(out.exit, 0);
        let (idx, body) = http.searches.lock().unwrap()[0].clone();
        assert_eq!(idx, "xc-kv-b1*", "STAR-direct glob, not the dash form");
        assert_eq!(
            body,
            serde_json::json!({
                "size": 5,
                "query": { "bool": { "must": [
                    { "multi_match": { "query": "addReplyNull",
                        "fields": ["body", "defs", "title"] } }
                ]}}
            }),
            "mapping-resolved fields (defs_expanded dropped); no highlight, no _source projection"
        );
        assert!(out
            .text
            .contains("─── valkey/src/networking.c:1460  (score 4.20, BSD-3-Clause)"));
        assert!(out.text.contains("function addReplyNull @ line 1"));
        assert!(out.text.contains("Cite file:line"));
        assert!(!out.is_error);
        // No field-report nudge marker was ever part of this port (dropped
        // 2026-09-18): the prose must not ask the agent to report back.
        assert!(!out.text.to_lowercase().contains("report back"));
    }

    #[test]
    fn restricted_licence_warns_under_the_passage_end_to_end() {
        let root = root_with_state();
        std::fs::write(
            root.join("corpora/kv/corpus.json"),
            "{\"corpus\":\"kv\",\"repos\":[{\"repo\":\"valkey\",\"url\":\"u\",\"licence\":\"AGPL\"}]}",
        )
        .unwrap();
        let http = FakeHttp::new();
        let out = run_code_query(
            &root,
            &http,
            "u",
            &CodeParams::new("kv", "q"),
            "`--stale-ok`",
        );
        assert!(out
            .text
            .contains("!! AGPL: adapt the APPROACH, do not copy the code"));
    }

    #[test]
    fn not_in_state_not_loaded_and_no_match_are_three_different_answers() {
        let root = root_with_state();

        // Not in the ledger at all: usage-class refusal naming the fix (exit 2).
        let out = run_code_query(
            &root,
            &FakeHttp::new(),
            "u",
            &CodeParams::new("ghost", "q"),
            "`--stale-ok`",
        );
        assert_eq!(out.exit, 2);
        assert!(out.text.contains("corpus 'ghost' is not indexed"));
        assert!(out.text.contains("xerj corpus index ghost"));

        // In the ledger, 0 live indices: the DISTINCT exit-3 diagnosis.
        let mut http = FakeHttp::new();
        http.live = vec![];
        let out = run_code_query(
            &root,
            &http,
            "http://localhost:9200",
            &CodeParams::new("kv", "q"),
            "`--stale-ok`",
        );
        assert_eq!(out.exit, 3, "not-loaded is its own exit code");
        assert!(out.text.contains("This is NOT a 'no match'"));

        // Loaded, zero hits: exit 1 with the verbatim fall-back prose.
        let mut http = FakeHttp::new();
        http.hits = vec![];
        let out = run_code_query(
            &root,
            &http,
            "u",
            &CodeParams::new("kv", "xyzzy plugh"),
            "`--stale-ok`",
        );
        assert_eq!(out.exit, 1);
        assert!(!out.is_error, "a miss is an answer, not an MCP error");
        assert!(
            out.text.contains(
                "The corpus is likely wrong for this task — fall back to normal work rather \
                 than forcing a bad match."
            ),
            "verbatim fall-back prose: {}",
            out.text
        );

        // --json with zero hits: still exit 1, empty array on stdout.
        let mut p = CodeParams::new("kv", "xyzzy plugh");
        p.as_json = true;
        let out = run_code_query(&root, &http, "u", &p, "`--stale-ok`");
        assert_eq!(out.exit, 1);
        assert!(out.json.is_some());
    }

    #[test]
    fn a_stale_index_is_refused_and_stale_ok_is_the_only_override() {
        let root = root_with_state();
        let old = (chrono::Utc::now() - chrono::Duration::days(31))
            .format(state::STAMP_FORMAT)
            .to_string();
        std::fs::write(
            root.join("state/kv.json"),
            format!(
                "{{\"corpus\":\"kv\",\"indexed_at\":\"{old}\",\"prefix\":\"xc-kv\",\
                 \"autoindex_exit\":0}}"
            ),
        )
        .unwrap();
        let out = run_code_query(
            &root,
            &FakeHttp::new(),
            "u",
            &CodeParams::new("kv", "q"),
            "`--stale-ok`",
        );
        assert_eq!(out.exit, 2);
        assert!(out.text.ends_with("or pass `--stale-ok`."));
        assert!(out.is_error);

        let mut p = CodeParams::new("kv", "q");
        p.stale_ok = true;
        let out = run_code_query(&root, &FakeHttp::new(), "u", &p, "`--stale-ok`");
        assert_eq!(out.exit, 0);
    }

    #[test]
    fn hybrid_is_one_native_top_level_request_with_capable_targeting() {
        let root = root_with_state();
        let mut http = FakeHttp::new();
        http.mapping = serde_json::json!({
            "xc-kv-b1-000": { "mappings": { "properties": { "body": { "type": "semantic_text" } } } },
            "xc-kv-b1-001": { "mappings": { "properties": { "body": { "type": "text" } } } }
        });
        let mut p = CodeParams::new("kv", "addReplyNull");
        p.mode = Mode::Hybrid;
        let out = run_code_query(&root, &http, "u", &p, "`--stale-ok`");
        assert_eq!(out.exit, 0);

        let reqs = http.searches.lock().unwrap();
        // [0] = BM25 preflight over the wildcard; [1] = the fused request,
        // aimed ONLY at the capable index (the plain-text sibling would 400
        // the whole wildcard).
        assert_eq!(reqs[0].0, "xc-kv-b1*");
        assert_eq!(reqs[1].0, "xc-kv-b1-000");
        let hybrid = &reqs[1].1;
        assert!(
            hybrid.pointer("/query/hybrid").is_some(),
            "ONE native top-level hybrid: {hybrid}"
        );
        assert_eq!(
            hybrid.pointer("/query/hybrid/fusion").unwrap(),
            &serde_json::json!({ "method": "rrf", "k": 60 })
        );
        assert_eq!(
            hybrid
                .pointer("/query/hybrid/queries")
                .unwrap()
                .as_array()
                .map(Vec::len),
            Some(2)
        );
        assert_eq!(hybrid.pointer("/size"), Some(&serde_json::json!(5)));
        // The arms-ran note says BOTH arms and the lexical-only exclusion.
        assert!(
            out.text
                .contains("[hybrid RRF(k=60) — BM25 over 2 index(es), vector over 1 of 2"),
            "{}",
            out.text
        );
        assert!(
            out.text.contains("rrf 4.20"),
            "hybrid scores render as rrf: {}",
            out.text
        );
    }

    #[test]
    fn hybrid_without_a_capable_index_degrades_to_bm25_and_says_so() {
        let root = root_with_state();
        let http = FakeHttp::new(); // mapping has plain text only
        let mut p = CodeParams::new("kv", "q");
        p.mode = Mode::Hybrid;
        let out = run_code_query(&root, &http, "u", &p, "`--stale-ok`");
        assert_eq!(out.exit, 0);
        assert!(
            out.text.contains("[BM25 only —"),
            "degradation is SAID: {}",
            out.text
        );
        assert_eq!(
            http.searches.lock().unwrap().len(),
            1,
            "no fused request was sent"
        );
    }

    #[test]
    fn hybrid_reports_an_honest_miss_when_the_bm25_preflight_is_empty() {
        let root = root_with_state();
        let mut http = FakeHttp::new();
        http.mapping = serde_json::json!({
            "xc-kv-b1-000": { "mappings": { "properties": { "body": { "type": "semantic_text" } } } }
        });
        http.hits = vec![]; // preflight AND any vector hit: nothing
        let mut p = CodeParams::new("kv", "xyzzy plugh");
        p.mode = Mode::Hybrid;
        let out = run_code_query(&root, &http, "u", &p, "`--stale-ok`");
        assert_eq!(
            out.exit, 1,
            "vector confidence must not launder a lexical miss"
        );
        assert!(
            out.text.contains("fall back to normal work"),
            "{}",
            out.text
        );
    }

    #[test]
    fn transport_failure_is_exit_2_never_fake_zero_indices() {
        let root = root_with_state();
        let mut http = FakeHttp::new();
        http.fail_search = true;
        let out = run_code_query(
            &root,
            &http,
            "u",
            &CodeParams::new("kv", "q"),
            "`--stale-ok`",
        );
        assert_eq!(out.exit, 2);
        assert!(out.text.contains("search failed"), "{}", out.text);
    }
}
