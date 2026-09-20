//! The local judge: System One wire-compatible decisions from a vote.
//!
//! Two surfaces, one mechanism:
//!
//! - `POST /v1/systemone` (native router) speaks TypeSafe's documented System
//!   One request/response shape, so a client written for their API —
//!   `jev-reranker`, the official SDKs — can point `TYPESAFE_ENDPOINT` at a
//!   XERJ node and keep working. Answers come from a weighted
//!   nearest-neighbour vote over a labelled-history index
//!   (`[decisions] index`), the retrieval analogue of a judge model that
//!   `benchmarks/decisions-as-retrieval` measures.
//! - `POST /_decide` (ES-compat router) is the same vote without the wire
//!   costume: it names its index per request and returns the neighbours,
//!   their labels and scores, and an abstain verdict.
//!
//! Nothing leaves the node. This module adds no outbound client; the vote is
//! an ordinary search against an ordinary index. That is the point of the
//! feature — the same interface, with the evidence staying home.
//!
//! # The wire contract, and where we deliberately break it
//!
//! Request: `{ state, model, questions: { id: { type, instructions, criteria } } }`.
//! `state` may be a string or an object; question `instructions` may be a
//! string, object or array (all string leaves are joined). Strings may point
//! at state data with backtick paths — `` `documents.doc_0` `` — which are
//! resolved against `state` before the vote text is built. That is the
//! reference pattern of the System One API and the shape `jev-reranker`
//! sends in its listwise mode.
//!
//! Response: `{ model, answers, usage }` with `answers` keyed exactly by the
//! question ids sent and each answer carrying only its documented fields —
//! clients parse these strictly. Extras ride at the top level (`decisions`),
//! which every verified client ignores.
//!
//! Two deliberate breaks, both documented here and in docs/RERANK.md:
//!
//! 1. `model` is echoed as `xerj-history-vote-1`, never as a Jev model name.
//!    Echoing `jev-1.13.0` would claim these probabilities are the hosted
//!    model's. The requested name is reported as `decisions.requested_model`.
//! 2. A question with no support — an empty history, or no labelled
//!    neighbour — is a 422 naming the question ids. Zero support must not
//!    become a fabricated 0.5. (`/_decide` says the same thing as `abstain`.)
//!
//! `score` questions (2–10 ordinal levels, weighted-average answer) have no
//! vote analogue and no benchmark: 422, in the docs.

use std::collections::BTreeMap;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};
use xerj_common::config::DecisionsConfig;
use xerj_query::parse_request;

use crate::state::AppState;

/// The model id this endpoint truthfully is. Never a Jev name.
pub const MODEL_ID: &str = "xerj-history-vote-1";

/// One question's vote text is bounded so a caller cannot make the node
/// search for a novel of its choosing; matches the rerank stage's philosophy
/// of caller-chosen cost with a server-side ceiling.
const MAX_VOTE_TEXT_CHARS: usize = 100_000;
/// The System One API documents at most 255 options per `choice`.
const MAX_CHOICE_OPTIONS: usize = 255;
/// Matches `rerank.max_window`: one request cannot ask for more questions
/// than the rerank stage would judge.
const MAX_QUESTIONS: usize = 300;

// ─────────────────────────────────────────────────────────────────────────────
// POST /v1/systemone
// ─────────────────────────────────────────────────────────────────────────────

pub async fn systemone(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let started = Instant::now();
    let cfg = state.config.decisions.clone();
    if cfg.index.is_empty() {
        return not_configured();
    }

    let questions = match body.get("questions") {
        Some(Value::Object(q)) if !q.is_empty() => q,
        _ => return unprocessable("`questions` must be a non-empty object"),
    };
    if questions.len() > MAX_QUESTIONS {
        return unprocessable(&format!(
            "`questions` must hold at most {MAX_QUESTIONS} questions (got {})",
            questions.len()
        ));
    }
    let state_val = body.get("state").cloned().unwrap_or(Value::Null);

    // Per question: the resolved vote text and the parsed type/criteria.
    let mut parsed: Vec<(String, Question)> = Vec::with_capacity(questions.len());
    for (id, q) in questions {
        match parse_question(id, q, &state_val) {
            Ok(question) => parsed.push((id.clone(), question)),
            Err(reason) => return unprocessable(&reason),
        }
    }

    // One vote per question. Collected first so a no-support question fails
    // the whole request — the wire has no per-question error field, and a
    // half-answered judgement history would look like a ranking.
    let mut answers = serde_json::Map::new();
    let mut evidence = serde_json::Map::new();
    let mut unsupported: Vec<String> = Vec::new();
    for (id, question) in &parsed {
        let neighbours = match vote_neighbours(&state, &cfg, &question.vote_text).await {
            Ok(n) => n,
            Err(resp) => return resp, // index missing / search failure: surfaced, not guessed
        };
        let labelled: Vec<(f64, &str)> = neighbours
            .iter()
            .map(|(w, label, _)| (*w, label.as_str()))
            .collect();
        if labelled.is_empty() {
            unsupported.push(id.clone());
            continue;
        }
        let total: f64 = labelled.iter().map(|(w, _)| *w).sum();
        let mut weight_by_label: BTreeMap<&str, f64> = BTreeMap::new();
        for (w, label) in &labelled {
            *weight_by_label.entry(label).or_insert(0.0) += *w;
        }
        let share = |label: &str| weight_by_label.get(label).copied().unwrap_or(0.0) / total;
        match &question.kind {
            Kind::Noul => {
                // Exactly the documented fields — clients parse answers
                // strictly and break on extras.
                answers.insert(
                    id.clone(),
                    json!({ "type": "noul", "noul": round6(share(&cfg.positive_label)) }),
                );
            }
            Kind::Choice { options } => {
                let criteria_total: f64 =
                    options.iter().map(|o| weight_by_label.get(o.as_str()).copied().unwrap_or(0.0)).sum();
                if criteria_total <= 0.0 {
                    unsupported.push(id.clone());
                    continue;
                }
                let mut probabilities = serde_json::Map::new();
                for option in options {
                    probabilities.insert(
                        option.clone(),
                        json!(round6(weight_by_label.get(option.as_str()).copied().unwrap_or(0.0) / total)),
                    );
                }
                let (winner, _) = options
                    .iter()
                    .map(|o| (o, weight_by_label.get(o.as_str()).copied().unwrap_or(0.0)))
                    .max_by(|(a, aw), (b, bw)| aw.partial_cmp(bw).unwrap().then(a.cmp(b)))
                    .expect("non-empty options");
                answers.insert(
                    id.clone(),
                    json!({
                        "type": "choice",
                        "choice": winner,
                        "confidence": round6(weight_by_label.get(winner.as_str()).copied().unwrap_or(0.0) / total),
                        "probabilities": Value::Object(probabilities),
                    }),
                );
            }
        }
        let (best_label, best_weight) = weight_by_label
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .expect("non-empty");
        evidence.insert(
            id.clone(),
            json!({
                "label": best_label,
                "support": round6(best_weight / total),
                "neighbours": labelled.len(),
                "found": neighbours.len(),
            }),
        );
    }
    if !unsupported.is_empty() {
        return no_support(&unsupported, &cfg);
    }

    let _ = state.metrics.record_query(&cfg.index, "systemone_vote", started.elapsed().as_secs_f64());
    Json(json!({
        "model": MODEL_ID,
        "answers": Value::Object(answers),
        "usage": { "input_tokens": 0, "output_tokens": 0 },
        "decisions": {
            "index": cfg.index,
            "k": cfg.k,
            "model": MODEL_ID,
            "requested_model": body.get("model").and_then(Value::as_str).unwrap_or(""),
            "evidence": Value::Object(evidence),
            "took_ms": started.elapsed().as_millis() as u64,
        }
    }))
    .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /v1/models
// ─────────────────────────────────────────────────────────────────────────────

/// The official SDKs list models before calling. One truthful entry.
pub async fn models(State(state): State<AppState>) -> axum::response::Response {
    let configured = !state.config.decisions.index.is_empty();
    Json(json!({
        "models": [{
            "name": MODEL_ID,
            "description": if configured {
                "Weighted nearest-neighbour vote over the configured decisions index — local, no egress"
            } else {
                "Weighted nearest-neighbour vote over a labelled-history index (no [decisions] index configured: /v1/systemone answers 503)"
            },
            "release_date": "2026-09-20",
        }]
    }))
    .into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /_decide
// ─────────────────────────────────────────────────────────────────────────────

/// The audit surface: the same vote, naming its index per request, returning
/// the neighbours it judged by, and abstaining instead of erroring.
pub async fn decide(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let started = Instant::now();
    let defaults = &state.config.decisions;
    let index = match body.get("index").and_then(Value::as_str) {
        Some(i) if !i.trim().is_empty() => i.to_string(),
        _ => return unprocessable("`index` must name the judgement-history index"),
    };
    let question = match body.get("question").and_then(Value::as_str) {
        Some(q) if !q.trim().is_empty() => q.to_string(),
        _ => return unprocessable("`question` must be a non-empty string"),
    };
    let k = body
        .get("k")
        .and_then(Value::as_u64)
        .map(|v| v.clamp(1, 100) as usize)
        .unwrap_or(defaults.k);
    let positive = body
        .get("positive_label")
        .and_then(Value::as_str)
        .unwrap_or(&defaults.positive_label)
        .to_string();
    let mut cfg = DecisionsConfig {
        index: index.clone(),
        k,
        positive_label: positive.clone(),
        ..defaults.clone()
    };
    cfg.label_field = defaults.label_field.clone();
    cfg.text_field = defaults.text_field.clone();

    let neighbours = match vote_neighbours(&state, &cfg, &clip(&question, MAX_VOTE_TEXT_CHARS)).await
    {
        Ok(n) => n,
        Err(resp) => return resp,
    };
    let total: f64 = neighbours.iter().map(|(w, _, _)| *w).sum();
    let mut weight_by_label: BTreeMap<String, f64> = BTreeMap::new();
    for (w, label, _) in &neighbours {
        *weight_by_label.entry(label.to_string()).or_insert(0.0) += *w;
    }
    let (label, confidence, abstain, reason) = match weight_by_label
        .iter()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    {
        Some((l, w)) if total > 0.0 => {
            let c = *w / total;
            let abstain = c < defaults.min_confidence;
            let reason = if abstain {
                Some(format!(
                    "confidence {:.3} below decisions.min_confidence {:.3}",
                    c, defaults.min_confidence
                ))
            } else {
                None
            };
            (Some(l.clone()), c, abstain, reason)
        }
        _ => (
            None,
            0.0,
            true,
            Some("no labelled neighbour in the history index".to_string()),
        ),
    };

    let neighbour_list: Vec<Value> = neighbours
        .iter()
        .map(|(w, label, hit)| {
            json!({
                "_id": hit.id,
                "label": label,
                "_score": hit.score,
                "weight": round6(*w),
                "text": hit.source.get(&cfg.text_field).cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    let _ = state
        .metrics
        .record_query(&index, "decide", started.elapsed().as_secs_f64());

    let mut resp = json!({
        "index": index,
        "k": k,
        "positive_label": positive,
        "label": label,
        "confidence": round6(confidence),
        "abstain": abstain,
        "neighbours": neighbour_list,
        "took_ms": started.elapsed().as_millis() as u64,
    });
    if let Some(r) = reason {
        resp["reason"] = json!(r);
    }
    Json(resp).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// The vote
// ─────────────────────────────────────────────────────────────────────────────

struct Question {
    vote_text: String,
    kind: Kind,
}

enum Kind {
    Noul,
    Choice { options: Vec<String> },
}

fn parse_question(id: &str, q: &Value, state_val: &Value) -> Result<Question, String> {
    let obj = q
        .as_object()
        .ok_or_else(|| format!("question `{id}` must be an object"))?;
    let kind = match obj.get("type").and_then(Value::as_str) {
        Some("noul") => Kind::Noul,
        Some("choice") => {
            let criteria = obj
                .get("criteria")
                .and_then(Value::as_object)
                .ok_or_else(|| format!("question `{id}` is a choice and needs `criteria` naming its options"))?;
            if criteria.is_empty() || criteria.len() > MAX_CHOICE_OPTIONS {
                return Err(format!(
                    "question `{id}`: `criteria` must name 1..{MAX_CHOICE_OPTIONS} options (got {})",
                    criteria.len()
                ));
            }
            Kind::Choice {
                options: criteria.keys().cloned().collect(),
            }
        }
        Some("score") => {
            return Err(format!(
                "question `{id}`: `score` questions (2-10 ordinal levels) have no vote analogue \
                 and are not served; split them into nouls or use /_decide"
            ))
        }
        other => {
            return Err(format!(
                "question `{id}`: `type` must be \"noul\" or \"choice\" (got {})",
                other.unwrap_or("<missing>")
            ))
        }
    };
    // `instructions` is polymorphic on this wire: string, object or array.
    // Every string leaf joins the vote text — an object's fields arrive in
    // serde_json's BTreeMap order, which is stable but arbitrary; the text is
    // retrieved by content, not by field order.
    let mut text = String::new();
    if let Some(instructions) = obj.get("instructions") {
        join_strings(instructions, &mut text);
    }
    // `criteria` descriptions are NOT embedded: the vote is over text the
    // history was labelled on, and no benchmark measures criteria text.
    let resolved = resolve_backticks(&text, state_val);
    // The query rides along from state — a plain-string state whole, an
    // object state via its `query` field, exactly the two shapes the wire's
    // own examples use. Other state fields are referenced (or not) by the
    // instructions' backtick paths, which have already been resolved.
    let mut vote_text = resolved;
    match state_val {
        Value::String(s) => {
            vote_text.push(' ');
            vote_text.push_str(s);
        }
        Value::Object(m) => {
            if let Some(q) = m.get("query").and_then(Value::as_str) {
                vote_text.push(' ');
                vote_text.push_str(q);
            }
        }
        _ => {}
    }
    let vote_text = clip(vote_text.trim(), MAX_VOTE_TEXT_CHARS);
    Ok(Question { vote_text, kind })
}

/// One weighted-neighbour record: (weight, label, hit). Unlabelled hits are
/// dropped here — they carry no vote.
async fn vote_neighbours(
    state: &AppState,
    cfg: &DecisionsConfig,
    text: &str,
) -> Result<Vec<(f64, String, xerj_query::executor::Hit)>, axum::response::Response> {
    let idx = match state.engine.get_index(&cfg.index) {
        Ok(i) => i,
        Err(e) => return Err(index_error(&cfg.index, &e.to_string())),
    };
    // The text field is configuration, not a literal — build the match by
    // hand so the field name is data, not syntax.
    let mut match_on = serde_json::Map::new();
    match_on.insert(cfg.text_field.clone(), json!(text));
    let query_body = json!({
        "query": { "match": Value::Object(match_on) },
        "size": cfg.k,
        "_source": true,
    });
    let search_req = match parse_request(&query_body) {
        Ok(r) => r,
        Err(e) => return Err(unprocessable(&format!("internal query would not parse: {e}"))),
    };
    let result = match idx.search(&search_req).await {
        Ok(r) => r,
        Err(e) => return Err(index_error(&cfg.index, &e.to_string())),
    };
    let mut out = Vec::with_capacity(result.hits.len());
    for (i, hit) in result.hits.into_iter().enumerate() {
        if let Some(label) = hit.source.get(&cfg.label_field).and_then(Value::as_str) {
            out.push((1.0 / (i as f64 + 1.0), label.to_string(), hit));
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Text helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Join every string leaf of a polymorphic `instructions` value.
fn join_strings(v: &Value, out: &mut String) {
    match v {
        Value::String(s) => {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(s);
        }
        Value::Array(a) => a.iter().for_each(|x| join_strings(x, out)),
        Value::Object(m) => m.values().for_each(|x| join_strings(x, out)),
        _ => {}
    }
}

/// Resolve `` `dotted.path` `` references against `state`. An unresolvable
/// path is left literally in place — visible in the vote text rather than
/// silently dropped.
fn resolve_backticks(text: &str, state: &Value) -> String {
    if !text.contains('`') || state.is_null() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('`') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('`') {
            Some(end) => {
                let path = &after[..end];
                match walk(state, path) {
                    Some(v) => out.push_str(v),
                    None => out.push_str(&rest[start..start + end + 2]),
                }
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&rest[start..]);
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

fn walk<'a>(v: &'a Value, path: &str) -> Option<&'a str> {
    let mut cur = v;
    for seg in path.split('.') {
        cur = cur.get(seg)?;
    }
    cur.as_str()
}

fn clip(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    s[..cut].to_string()
}

fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Error bodies — every one names its cause and never guesses an answer
// ─────────────────────────────────────────────────────────────────────────────

fn error_body(status: u16, kind: &str, reason: String) -> axum::response::Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::UNPROCESSABLE_ENTITY),
        Json(json!({ "error": { "type": kind, "reason": reason } })),
    )
        .into_response()
}

fn unprocessable(reason: &str) -> axum::response::Response {
    error_body(422, "illegal_argument", reason.to_string())
}

fn not_configured() -> axum::response::Response {
    error_body(
        503,
        "not_configured",
        "no decisions index is configured: set [decisions] index to an index of labelled \
         examples (one document per example, with the label and text fields this node's \
         decisions settings name), then POST /v1/systemone again"
            .to_string(),
    )
}

fn index_error(index: &str, cause: &str) -> axum::response::Response {
    error_body(
        422,
        "history_index_error",
        format!("decisions index `{index}` could not be searched: {cause}"),
    )
}

fn no_support(ids: &[String], cfg: &DecisionsConfig) -> axum::response::Response {
    error_body(
        422,
        "no_support",
        format!(
            "no labelled neighbour in `{}` for question{} {} (k={}); zero support is an \
             error, not a fabricated probability — add labelled examples to the history \
             index or use POST /_decide to inspect what is there",
            cfg.index,
            cfg.k,
            if ids.len() == 1 { "" } else { "s" },
            ids.iter().map(|i| format!("`{i}`")).collect::<Vec<_>>().join(", ")
        ),
    )
}
