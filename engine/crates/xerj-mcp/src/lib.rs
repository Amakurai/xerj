//! `xerj-mcp` — a Model Context Protocol (MCP) server for XERJ.
//!
//! This crate is both a **library** and a standalone binary. The library entry
//! point is [`run`], which the main `xerj` binary calls for its `xerj mcp`
//! subcommand — so a user who installed the single `xerj` binary already has
//! the MCP server, with nothing to clone and nothing to compile. The
//! standalone `xerj-mcp` binary is a thin wrapper over the same [`run`] and
//! stays available for CI and for embedding without the engine.
//!
//! It speaks the MCP **stdio transport**: newline-delimited
//! JSON-RPC 2.0 messages on stdin/stdout. It exposes XERJ to any MCP-capable
//! agent host (Claude Desktop, IDE agents, custom orchestrators) as eleven
//! tools that map 1:1 onto XERJ's real, verified REST surface. Every tool is
//! a *thin proxy*: it constructs exactly the request the running engine
//! already accepts and forwards it to a configurable base URL.
//!
//! ## The eleven canonical agent operations
//!
//! | MCP tool               | XERJ endpoint                         | Real capability |
//! |------------------------|---------------------------------------|-----------------|
//! | `xerj_search`          | `POST /{index}/_search`               | ES query-DSL search (full-text / keyword / structured) |
//! | `xerj_semantic_search` | `POST /{index}/_search` (`semantic`)  | server-side lexical embedding, no external key |
//! | `xerj_vector_search`   | `POST /{index}/_search` (`knn`)       | kNN over a `dense_vector` field (HNSW-served unfiltered, exact filtered) |
//! | `xerj_hybrid_search`   | `POST /{index}/_search` (`hybrid`)    | RRF or linear fusion of sub-queries |
//! | `xerj_memory_store`    | `POST /_memory/{ns}`                  | namespaced agent-memory write |
//! | `xerj_memory_recall`   | `POST /_memory/{ns}/_recall`          | recall by meaning (BM25 / semantic / vector), optionally graph-coupled |
//! | `xerj_brain_ego`       | `GET /_graph/{brain}/ego`             | one node's evidence-backed, bi-temporal link neighborhood |
//! | `xerj_brain_link`      | `POST /_graph/{brain}/link`           | assert a link (idempotent, deterministic edge_id) |
//! | `xerj_brain_unlink`    | `DELETE /_graph/{brain}/link/{id}`    | retire a link (soft-invalidate; never deletes) |
//! | `xerj_brain_overview`  | `GET /_graph/{brain}/overview`        | whole-brain orientation: counts, hubs, detectors, timeline |
//!
//! ## Honesty notes (must match the engine, never oversell)
//!
//! * Unfiltered kNN is **HNSW-served with exact rescoring** (measured recall@10
//!   1.00 on the official bench query); filtered kNN and other ineligible shapes
//!   (non-cosine, SQ8, small indexes) run the exact brute-force scan. The tool
//!   description says so.
//! * `rerank` on `xerj_search` / `xerj_hybrid_search` is **opt-in and needs a
//!   provider the node's operator configured**; this server cannot supply the
//!   key. It is also the one argument that makes the node send document text to
//!   a third party, and the tool description says so in those words. Ranking
//!   quality with the real provider model has **not been verified by this
//!   project** — the description promises the mechanism (a calibrated
//!   probability replaces `_score`), never an accuracy figure.
//! * `hybrid` supports `fusion: "rrf"` and `"linear"` only. `"learned"` is
//!   forwarded verbatim and the engine rejects it loudly; the schema therefore
//!   advertises only `rrf`/`linear`.
//! * The `xerj_brain_*` tools work a **link index, not a graph database**:
//!   deterministic lexical detection plus explicit assertions — no neural
//!   understanding, no query language, no unbounded traversal. Every brain
//!   tool description says both things, and engine refusals (e.g. the hops
//!   cap) are passed back to the agent verbatim.
//! * This proxy adds **no** capabilities of its own — whatever the engine
//!   returns (including errors) is passed straight back to the agent.
//!
//! ## Configuration (environment)
//!
//! * `XERJ_URL`  — base URL of the XERJ ES-compatible listener. Default
//!   `http://localhost:9200`.
//! * `XERJ_AUTH` — optional; if set, sent verbatim as the `Authorization`
//!   header on every proxied request (e.g. `ApiKey <token>`).
//!
//! When neither `XERJ_AUTH` nor `--auth` is given **and** the target is
//! loopback, the server falls back to the admin key a local node wrote for
//! itself (`<data_dir>/admin.key` — the same discovery `xerj autoindex`
//! uses) and announces the file it used on stderr. Auth is on by default, so
//! this fallback is what makes the documented `xerj init` one-command setup
//! work against a default node instead of 401-ing on every tool call (#961).
//! It never fires for a non-loopback URL: a locally readable key sent to
//! another host is a credential leak, not a convenience.
//!
//! Both have a flag form — `--url` and `--auth` — which wins over the
//! environment. MCP client configs usually set the `env` block, so the
//! environment stays the documented path; the flags exist for hosts that only
//! let you supply `args`.
//!
//! Diagnostics go to **stderr**; stdout is reserved exclusively for the
//! JSON-RPC stream.

use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// MCP protocol revision we default to when the client does not request one.
const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";
/// Default XERJ ES-compatible endpoint.
const DEFAULT_XERJ_URL: &str = "http://localhost:9200";

/// Shared per-process state: the HTTP client + where to proxy to.
struct Ctx {
    client: reqwest::Client,
    base_url: String,
    auth: Option<String>,
}

/// `--help` text for both entry points (`xerj mcp --help` and `xerj-mcp --help`).
pub const HELP_BODY: &str = "\
xerj mcp — Model Context Protocol (MCP) stdio server for XERJ

USAGE:
    xerj mcp [OPTIONS]              (or the standalone binary: xerj-mcp [OPTIONS])

Speaks MCP over stdio (newline-delimited JSON-RPC 2.0 on stdin/stdout) and
proxies eleven tools — xerj_search, xerj_semantic_search, xerj_vector_search,
xerj_hybrid_search, xerj_memory_store, xerj_memory_recall, xerj_brain_ego,
xerj_brain_link, xerj_brain_unlink, xerj_brain_overview, xerj_code_search — to a
XERJ node that
is ALREADY RUNNING. This command does not start a node; start one first with
`xerj --data-dir ./data`.

OPTIONS:
    --url <URL>     Base URL of the XERJ ES-compatible listener.
                    Overrides XERJ_URL. Default: http://localhost:9200
    --auth <VALUE>  Authorization header sent verbatim on every proxied
                    request (e.g. 'ApiKey <token>'). Overrides XERJ_AUTH.
    -h, --help      Show this help
    -V, --version   Print version and exit
    --disable-feedback
                    Do not print the feedback invitation above. Honoured in any
                    position, including after --help.
                    Env: XERJ_DISABLE_FEEDBACK=true

ENVIRONMENT:
    XERJ_URL        Same as --url
    XERJ_AUTH       Same as --auth

MCP CLIENT CONFIG (Claude Desktop, IDE agents, any mcpServers-shaped config):

    {
      \"mcpServers\": {
        \"xerj\": {
          \"command\": \"/home/you/.local/bin/xerj\",
          \"args\": [\"mcp\"],
          \"env\": { \"XERJ_URL\": \"http://localhost:9200\" }
        }
      }
    }

Use an ABSOLUTE path — `command -v xerj` prints it, and the installer's
default is ~/.local/bin/xerj. An MCP host launched from a desktop icon does
not inherit your shell's PATH, so a bare \"xerj\" fails there with no useful
error. When the node is not running --insecure, add
\"XERJ_AUTH\": \"ApiKey <key>\" beside XERJ_URL; the key is <data-dir>/admin.key.
For a loopback node (localhost / 127.0.0.1 / [::1]) that step is optional:
with no XERJ_AUTH given, the server reads <data-dir>/admin.key itself (the
same locations `xerj autoindex` checks) and says which file on stderr.

Diagnostics go to stderr; stdout carries only the JSON-RPC stream.
";

/// The help text with the feedback invitation spliced in above `USAGE:`, the
/// same position the other four surfaces use.
///
/// `xerj mcp` became a help surface after the invitation shipped, so by this
/// feature's own rule — every screen the top-level help advertises should ask —
/// it belongs here too.
pub fn help_text(feedback: bool) -> String {
    let (first, rest) = HELP_BODY
        .split_once("\nUSAGE:")
        .expect("HELP_BODY carries a USAGE: section");
    format!(
        "{first}\n{}USAGE:{rest}",
        xerj_common::feedback::block(feedback)
    )
}

/// Resolve the `Authorization` header value every proxied request will carry.
///
/// Precedence: `--auth` > `XERJ_AUTH` >, for a loopback base URL only, the
/// admin key a local node wrote for itself. The fallback exists because auth
/// is ON by default and `xerj init` deliberately writes no credential into
/// `.mcp.json` (project scope, meant to be committed) — without it, the
/// documented one-command setup 401s on every tool call (#961).
///
/// Returns the header value plus, when the value came from disk, the file it
/// came from — the caller announces that path on stderr so the fallback is
/// never silent (stdout stays JSON-RPC-only).
///
/// The loopback guard is the security boundary: discovery only ever runs for
/// a URL whose host is this machine, so a locally readable key can never be
/// sent off-box (including via the `http://localhost:9200@evil.com/` userinfo
/// trick — see `xerj_common::localauth`).
fn resolve_auth(
    auth_override: Option<String>,
    env_auth: Option<String>,
    base_url: &str,
) -> (Option<String>, Option<std::path::PathBuf>) {
    // Empty strings count as unset: an empty XERJ_AUTH in a committed
    // .mcp.json must not shadow discovery and turn into `Authorization: `.
    if let Some(explicit) = auth_override
        .filter(|s| !s.is_empty())
        .or_else(|| env_auth.filter(|s| !s.is_empty()))
    {
        return (Some(explicit), None);
    }
    if !xerj_common::localauth::url_is_loopback(base_url) {
        return (None, None);
    }
    match xerj_common::localauth::discover_local_admin_key() {
        // `XERJ_AUTH` is a complete header value ("ApiKey <token>"), so the
        // discovered bare key is formatted to the same documented shape.
        Some((key, path)) => (Some(format!("ApiKey {key}")), Some(path)),
        None => (None, None),
    }
}

/// Library entry point. `args` are the arguments *after* the subcommand
/// (`xerj mcp <args>`) or after the program name (`xerj-mcp <args>`).
///
/// Runs until stdin reaches EOF, which is how an MCP host shuts a stdio
/// server down.
pub async fn run(args: &[String]) -> anyhow::Result<()> {
    let mut url_override: Option<String> = None;
    let mut auth_override: Option<String> = None;

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{}", help_text(xerj_common::feedback::enabled()));
                return Ok(());
            }
            // Accepted, not acted on here: `feedback::enabled` reads the whole
            // argument list, so the flag works in any position. Without this
            // arm it would be rejected as an unknown argument on the one
            // surface it is meant to control.
            xerj_common::feedback::DISABLE_FLAG => {}
            "-V" | "--version" => {
                println!("xerj-mcp {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--url" => {
                url_override = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--url requires a value"))?
                        .clone(),
                );
            }
            "--auth" => {
                auth_override = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--auth requires a value"))?
                        .clone(),
                );
            }
            other => {
                // Fail loudly: a typo'd flag that is silently ignored leaves the
                // agent talking to the wrong node with no way to notice.
                anyhow::bail!("unknown argument `{other}` — see `xerj mcp --help`");
            }
        }
    }

    let base_url = url_override
        .or_else(|| std::env::var("XERJ_URL").ok())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_XERJ_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let (auth, auth_key_file) =
        resolve_auth(auth_override, std::env::var("XERJ_AUTH").ok(), &base_url);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    let ctx = Ctx {
        client,
        base_url,
        auth,
    };

    eprintln!(
        "xerj-mcp v{} — MCP stdio server, proxying to {}",
        env!("CARGO_PKG_VERSION"),
        ctx.base_url
    );

    // Announced, never silent (same rule as `xerj autoindex`): MCP hosts
    // spawn this server with an arbitrary cwd, so which `admin.key` the
    // relative candidates resolved to is exactly the thing an onboarding
    // user needs to see when it works here and not there.
    if let Some(path) = auth_key_file {
        eprintln!(
            "xerj-mcp: no --auth/XERJ_AUTH given; using the admin key at {}",
            path.display()
        );
    }

    // stdout is the JSON-RPC channel; stderr is for logs only.
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parsed: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                // Parse error: reply per JSON-RPC (id unknown → null).
                let resp = rpc_error(Value::Null, -32700, format!("parse error: {e}"));
                write_msg(&mut stdout, &resp).await?;
                continue;
            }
        };

        // JSON-RPC allows a batch (array) of messages. MCP 2024-11-05 uses it;
        // 2025-06-18 dropped it. Support both: object → single, array → batch.
        let responses: Vec<Value> = match parsed {
            Value::Array(batch) => {
                let mut out = Vec::new();
                for m in batch {
                    if let Some(r) = handle_message(&ctx, m).await {
                        out.push(r);
                    }
                }
                out
            }
            other => handle_message(&ctx, other).await.into_iter().collect(),
        };

        for resp in responses {
            write_msg(&mut stdout, &resp).await?;
        }
    }

    Ok(())
}

/// Serialize one JSON-RPC message and write it as a single newline-terminated
/// line (the MCP stdio framing), then flush.
async fn write_msg(stdout: &mut tokio::io::Stdout, msg: &Value) -> anyhow::Result<()> {
    let mut buf = serde_json::to_vec(msg)?;
    buf.push(b'\n');
    stdout.write_all(&buf).await?;
    stdout.flush().await?;
    Ok(())
}

/// Route one JSON-RPC message. Returns `Some(response)` for requests and
/// `None` for notifications (no `id`) and messages that need no reply.
async fn handle_message(ctx: &Ctx, msg: Value) -> Option<Value> {
    let id = msg.get("id").cloned();
    let is_notification = id.is_none();
    let method = msg.get("method").and_then(Value::as_str)?.to_string();

    match method.as_str() {
        "initialize" => Some(rpc_result(id, initialize_result(&msg))),

        // Lifecycle / keepalive notifications — no response.
        "notifications/initialized" | "initialized" | "notifications/cancelled" => None,

        "ping" => Some(rpc_result(id, json!({}))),

        "tools/list" => Some(rpc_result(id, json!({ "tools": tool_specs() }))),

        "tools/call" => Some(rpc_result(id, call_tool(ctx, &msg).await)),

        // Unknown method: error for requests, silence for notifications.
        _ => {
            if is_notification {
                None
            } else {
                Some(rpc_error(
                    id.unwrap_or(Value::Null),
                    -32601,
                    format!("method not found: {method}"),
                ))
            }
        }
    }
}

/// Build the `initialize` result, echoing the client's requested protocol
/// version when present (our JSON-RPC handling is version-agnostic).
fn initialize_result(msg: &Value) -> Value {
    let pv = msg
        .get("params")
        .and_then(|p| p.get("protocolVersion"))
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_PROTOCOL_VERSION);

    json!({
        "protocolVersion": pv,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": {
            "name": "xerj-mcp",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "instructions":
            "XERJ tools proxy to a running XERJ engine (ES-compatible). Use \
             xerj_search for retrieval — pass `query` as a plain string for \
             definition-first code search (ranks the defining file first, \
             returns the matching passage instead of whole files), or as an \
             ES query-DSL object for structured queries, \
             xerj_semantic_search for meaning-based recall over a semantic_text \
             field (embedding is server-side, no key), xerj_vector_search for \
             kNN over a dense_vector field, xerj_hybrid_search to fuse \
             lexical + vector results (rrf|linear), and xerj_memory_store / \
             xerj_memory_recall for durable agent memory recalled by meaning. \
             xerj_search and xerj_hybrid_search take an optional `rerank` \
             argument: a second stage that re-judges the top hits with an \
             external relevance model. It only works if the node's operator \
             configured a provider key, and it sends the returned document \
             text off the machine — leave it out unless first-stage order is \
             the problem and the user is fine with that. \
             The xerj_brain_* tools work a second brain (a bi-temporal, \
             evidence-carrying link index built by `xerj brain <folder>`): \
             xerj_brain_overview to orient, then xerj_brain_ego for one node's \
             neighborhood, then xerj_brain_link / xerj_brain_unlink to assert \
             or retire links — retired links stay replayable via as_of.",
    })
}

// ─────────────────────────── JSON-RPC helpers ──────────────────────────────

fn rpc_result(id: Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": result })
}

fn rpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() },
    })
}

/// A `tools/call` result carrying a single text block.
fn tool_text(text: impl Into<String>, is_error: bool) -> Value {
    json!({
        "content": [ { "type": "text", "text": text.into() } ],
        "isError": is_error,
    })
}

// ─────────────────────────────── Tools ─────────────────────────────────────

/// The two honesty sentences every `xerj_brain_*` tool description MUST carry
/// (SECOND_BRAIN_SPEC honesty rule; mirrors the crate's kNN precedent above).
const BRAIN_HONESTY: &str = " Honesty: links come from deterministic lexical \
     detection plus explicit assertions — not neural understanding. XERJ is a \
     search engine with a graph-shaped index over its own documents, not a \
     graph database.";

/// JSON-schema fragment for the optional `rerank` argument, shared by
/// `xerj_search` and `xerj_hybrid_search` so the two descriptions cannot drift.
///
/// The description is written for the reader that will act on it — an agent
/// deciding whether to spend a paid third-party call. It therefore says three
/// things a schema normally would not: when the stage is worth it, that it
/// needs a provider only the node's operator can configure, and that it is the
/// only search-time feature that sends document text off the node.
fn rerank_arg_schema(query_required: bool) -> Value {
    let defaults = xerj_rerank::RerankConfig::default();
    let question = if query_required {
        "`query` (string, REQUIRED for this tool): the natural-language question to \
         judge documents against."
    } else {
        "`query` (string): the natural-language question to judge documents against. \
         Filled in for you when the tool's `query` is a plain string (that mode also \
         returns the matching passage, which leads the text the judge reads); REQUIRED when \
         the tool's `query` is a `bool`, a `knn`, or anything else with no single \
         question in it (the node refuses rather than guess)."
    };
    let description = format!(
        "Optional second-stage reranking. XERJ sends the text of the top `window` hits \
         plus the question to an external relevance model, which returns a calibrated \
         probability (0..1) per document. Hits are reordered by it, that probability \
         REPLACES `_score`, and `min_score` prunes by it — an absolute relevance cut, \
         which a BM25 score cannot give you. \
         WORTH IT WHEN: a natural-language question over prose where the right document \
         is probably in the top {default_window} but not at the top; or you need a yes/no relevance \
         threshold rather than a relative rank. \
         NOT WORTH IT FOR: exact identifier or symbol lookups (the first stage already \
         ranks the definition first), filters, sorted or aggregation-only requests — it \
         adds a paid network round-trip per search. \
         NEEDS A CONFIGURED PROVIDER: the node's operator must have set a provider key \
         (`[rerank] api_key` or TYPESAFE_API_KEY). You cannot supply one in this call. \
         Without it the search fails with HTTP 503 `rerank_exception`; do not retry, \
         repeat the search without `rerank`. HTTP 403 means the operator disabled it. \
         PRIVACY: this is the only search-time feature that sends document text off the \
         node. Proxy embeddings (`[embedding] default_endpoint`) and the WAL tap also send \
         text off the node when an operator configures them, so never tell a user that \
         nothing else leaves the machine — do not use rerank on data that must stay local \
         unless the user has agreed. Only fields \
         the response returns are sent, so `_source` filtering also limits what leaves. \
         Pass true or {{}} for defaults, or an object with: {question} \
         `window` (int, default {default_window}, max {max_window}): how many top hits \
         are judged — every one is a paid judgement; `from`+`size` must fit inside it. \
         `min_score` (0..1): drop hits judged below this probability. \
         `fields` (string[]): which returned fields to send; default is every returned \
         string field. `instructions` (string): override what \"relevant\" means for \
         this corpus. `timeout_ms` (int, default {default_timeout_ms}). `model` (string). \
         Cannot be combined with `sort`. \
         READ THE RESPONSE'S `_rerank` BLOCK: `applied: true` means the order is the \
         judge's and every `_score` is a probability or null (a hit with no verdict, \
         counted in `unjudged`, sorts last); `applied: false` means the provider missed \
         the deadline or answered nothing usable, and you are looking at the engine's own \
         order with engine scores, stated in `reason`.",
        default_window = defaults.window,
        max_window = xerj_rerank::MAX_WINDOW,
        default_timeout_ms = defaults.timeout.as_millis(),
    );
    json!({ "type": ["object", "boolean"], "description": description })
}

/// The ten tool specifications advertised via `tools/list`. Input schemas are
/// plain JSON Schema; every property maps onto a field the engine accepts.
///
/// Public because this is the single source of truth for the published
/// agent-facing schema at `landing/docs/agents/schemas/mcp-tools.json`, which
/// `tests/published_schema_drift.rs` pins to this function. The published file
/// once advertised six tools while the binary served ten — nobody noticed,
/// because nothing compared them.
pub fn tool_specs() -> Value {
    json!([
        {
            "name": "xerj_search",
            "description":
                "Full-text / keyword / structured search over a XERJ index. Proxies \
                 POST /{index}/_search. Pass `query` as a PLAIN STRING for the \
                 definition-first code search (ranks the file that defines a symbol \
                 above files that merely mention it — the right default for code), \
                 or as an ES query-DSL object (e.g. {\"term\":{\"status\":\"open\"}}, \
                 or a bool clause) for structured queries. Omit `query` for match_all. \
                 Optional `rerank` adds a second stage that re-judges the top hits with \
                 an external relevance model — read that argument's description before \
                 using it: it needs a provider the operator configured, and it sends \
                 document text off the machine.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "string",
                        "description": "Index name or pattern to search. Defaults to `ax-*` — \
                         the prefix `xerj autoindex` writes — so the common case needs \
                         no argument. Pass an explicit name to search anything else."
                    },
                    "query": {
                        "type": ["object", "string"],
                        "description": "Plain string (definition-first code search) or ES query-DSL clause. Omit for match_all.",
                    },
                    "size": { "type": "integer", "description": "Max hits to return (default engine value)." },
                    "from": { "type": "integer", "description": "Offset for pagination." },
                    "sort": { "description": "ES sort clause (array or object)." },
                    "_source": { "description": "Source filtering (bool, field, or {includes,excludes})." },
                    "rerank": rerank_arg_schema(false)
                }
            }
        },
        {
            "name": "xerj_semantic_search",
            "description":
                "Meaning-based search over a `semantic_text` field. The query text is \
                 embedded SERVER-SIDE by XERJ's built-in lexical embedder (no external \
                 API key), then matched by vector similarity. Proxies POST /{index}/_search \
                 with {\"query\":{\"semantic\":{...}}}.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": { "type": "string" },
                    "field": { "type": "string", "description": "Name of the semantic_text field." },
                    "query": { "type": "string", "description": "Natural-language query text to embed and match." },
                    "k": { "type": "integer", "description": "Number of nearest results (default 10)." },
                    "filter": { "type": "object", "description": "Optional ES query clause applied as a pre-filter." }
                },
                "required": ["index", "field", "query"]
            }
        },
        {
            "name": "xerj_vector_search",
            "description":
                "K-nearest-neighbour search over a `dense_vector` field, given a \
                 caller-supplied query vector. NOTE: unfiltered kNN is HNSW-served \
                 (approximate) with exact rescoring — measured recall@10 1.00 on the \
                 official bench query; num_candidates sets the beam width (floored at \
                 800). Filtered kNN, non-cosine metrics, SQ8 fields, and small indexes \
                 run an exact brute-force scan. \
                 Proxies POST /{index}/_search with a top-level {\"knn\":{...}}.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": { "type": "string" },
                    "field": { "type": "string", "description": "Name of the dense_vector field." },
                    "query_vector": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Query embedding; length must match the field's dims."
                    },
                    "k": { "type": "integer", "description": "Number of nearest neighbours (default 10)." },
                    "num_candidates": { "type": "integer", "description": "Optional candidate pool size." },
                    "filter": { "type": "object", "description": "Optional ES query clause applied as a pre-filter." }
                },
                "required": ["index", "field", "query_vector"]
            }
        },
        {
            "name": "xerj_hybrid_search",
            "description":
                "Hybrid search: fuse several sub-queries (e.g. a lexical `match` plus a \
                 vector `knn`) into one ranked list. Fusion is `rrf` (reciprocal-rank) or \
                 `linear` (weighted). Proxies POST /{index}/_search with \
                 {\"query\":{\"hybrid\":{\"queries\":[...],\"fusion\":...}}}. \
                 Optional `rerank` re-judges the fused top hits with an external \
                 relevance model; here `rerank.query` is REQUIRED, because a hybrid \
                 search has several sub-queries and no single question to read.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": { "type": "string" },
                    "queries": {
                        "type": "array",
                        "description":
                            "Sub-queries to fuse. Each item is {\"query\": <ES query clause>, \
                             \"weight\": <number, optional>}.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": { "type": "object" },
                                "weight": { "type": "number" }
                            },
                            "required": ["query"]
                        }
                    },
                    "fusion": {
                        "type": "string",
                        "enum": ["rrf", "linear"],
                        "description": "Fusion strategy (default rrf)."
                    },
                    "size": { "type": "integer", "description": "Max fused hits to return." },
                    "rerank": rerank_arg_schema(true)
                },
                "required": ["index", "queries"]
            }
        },
        {
            "name": "xerj_memory_store",
            "description":
                "Store a durable agent memory in a namespace. The text is BM25-indexed \
                 and (via a semantic_text field) auto-embedded so it can later be recalled \
                 by meaning. Proxies POST /_memory/{namespace}. Set `dedup:true` to skip \
                 writing a near-identical existing memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string", "description": "Memory namespace (isolates recall)." },
                    "text": { "type": "string", "description": "Free text of the memory." },
                    "metadata": { "type": "object", "description": "Optional structured metadata." },
                    "id": { "type": "string", "description": "Optional explicit id (upsert)." },
                    "vector": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Optional precomputed embedding to enable vector recall."
                    },
                    "dedup": { "type": "boolean", "description": "Skip write if a near-duplicate exists." },
                    "dedup_threshold": { "type": "number", "description": "Similarity threshold for dedup." }
                },
                "required": ["namespace", "text"]
            }
        },
        {
            "name": "xerj_memory_recall",
            "description":
                "Recall the most relevant memories from a namespace. Default is BM25 text \
                 recall; set `semantic:true` to embed the query server-side and recall by \
                 meaning; set `hybrid:true` to run both over the same `query` and fuse them \
                 by reciprocal rank; supply `vector` for pure vector recall. Proxies \
                 POST /_memory/{namespace}/_recall.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" },
                    "query": { "type": "string", "description": "Query text (BM25, or embedded when semantic:true)." },
                    "semantic": { "type": "boolean", "description": "Embed `query` server-side and recall by meaning." },
                    "hybrid": {
                        "type": "boolean",
                        "description":
                            "Run BM25 and server-side semantic recall over `query` and fuse the \
                             two rankings (reciprocal rank fusion). Off by default; cannot be \
                             combined with `vector` or `semantic`."
                    },
                    "fusion": {
                        "type": "string",
                        "enum": ["rrf", "linear"],
                        "description": "Fusion for hybrid:true — `rrf` (default) or `linear`."
                    },
                    "vector": {
                        "type": "array",
                        "items": { "type": "number" },
                        "description": "Query embedding for pure vector recall (takes precedence)."
                    },
                    "k": { "type": "integer", "description": "Number of memories to return (default 10)." },
                    "filter": { "type": "object", "description": "Optional metadata pre-filter (ES query clause)." },
                    "recency_weight": {
                        "type": "number",
                        "description": "Blend relevance with recency in [0,1]; 0=pure relevance, 1=pure recency."
                    },
                    "graph": {
                        "type": "object",
                        "description":
                            "Optional graph coupling with the namespace's second-brain links \
                             (edges index `.xerj-memory-{ns}-edges`). `restrict` recalls only \
                             within graph reach of the seeds; `blend` lets graph proximity pull \
                             related memories up. Degrades gracefully (`no_edges_index:true` in \
                             the response) when the namespace has no links yet.",
                        "properties": {
                            "mode": { "type": "string", "enum": ["restrict", "blend"] },
                            "seeds": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Seed node ids. Required for restrict; blend defaults to the top-5 base-recall hits."
                            },
                            "hops": { "type": "integer", "description": "1 (default) or 2." },
                            "types": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Link-type allowlist."
                            },
                            "weight": { "type": "number", "description": "Blend weight in [0,1], default 0.3. Ignored for restrict." },
                            "as_of": { "description": "Bi-temporal cut: epoch-ms number or RFC3339 string (default now)." }
                        },
                        "required": ["mode"]
                    }
                },
                "required": ["namespace"]
            }
        },
        {
            "name": "xerj_brain_ego",
            "description": format!(
                "One node's annotated neighborhood in a second brain: every link with \
                 direction, hop, type, confidence, and the evidence that justified it, \
                 plus node previews and a `not_shown` accounting of anything clipped. \
                 Links are believed-since/retired (`valid_at`/`invalid_at`); pass `as_of` \
                 to replay a past moment. Hops cap at 2 by design — iterate from \
                 `reachable`. Proxies GET /_graph/{{brain}}/ego.{BRAIN_HONESTY}"),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "brain": { "type": "string", "description": "Brain name (as created by `xerj brain <folder>` or the link tool)." },
                    "node": { "type": "string", "description": "Node id to expand from (a document _id in the brain's nodes index)." },
                    "hops": { "type": "integer", "enum": [1, 2], "description": "Neighborhood radius: 1 (default) or 2. Capped at 2 by design." },
                    "direction": { "type": "string", "enum": ["out", "in", "both"], "description": "Which links to follow (default both)." },
                    "types": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Link-type allowlist (e.g. [\"references\"]); absent = all types."
                    },
                    "limit": { "type": "integer", "description": "Max returned links; engine clamps to 1..=1000 (default 100). Clipped remainder is reported in `not_shown`." },
                    "as_of": { "description": "Bi-temporal cut: epoch-ms number or RFC3339 string. Replays what the brain believed at that moment (default now)." },
                    "include_expired": { "type": "boolean", "description": "Also return retired links; their `invalid_at` says when they stopped being believed (default false)." },
                    "include_nodes": { "type": "boolean", "description": "Hydrate node title/preview summaries (default true here — the proxy sets it explicitly)." }
                },
                "required": ["brain", "node"]
            }
        },
        {
            "name": "xerj_brain_link",
            "description": format!(
                "Assert a link between two nodes in a second brain, with the evidence \
                 that justifies it — pass the exact quote you are relying on. \
                 edge_id is deterministic over (src, type, dst, valid_at), so \
                 re-asserting the same fact is idempotent (`created:true/false`); \
                 `valid_at` defaults to server-now, so pass it explicitly when a \
                 retry must dedupe. Proxies POST \
                 /_graph/{{brain}}/link.{BRAIN_HONESTY}"),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "brain": { "type": "string", "description": "Brain name. The edges index is created lazily on first link." },
                    "src": { "type": "string", "description": "Source node id (a document _id)." },
                    "dst": { "type": "string", "description": "Destination node id. Must differ from `src` (no self-links)." },
                    "type": { "type": "string", "description": "Link type, e.g. references, mentions, contradicts." },
                    "evidence": {
                        "type": "object",
                        "description": "Why this link exists: {quote, source, offset} — the exact text relied on, where it came from, and its byte offset. Optional, but a link without evidence is shown as asserted-not-detected.",
                        "properties": {
                            "quote": { "type": "string" },
                            "source": { "type": "string" },
                            "offset": { "type": "integer" }
                        }
                    },
                    "weight": { "type": "number", "description": "Link strength in [0,1], default 1.0." },
                    "confidence": { "type": "number", "description": "Assertion confidence in [0,1], default 1.0." },
                    "valid_at": { "description": "When the fact became true: epoch-ms number or RFC3339 string (default now). Part of the deterministic edge_id." }
                },
                "required": ["brain", "src", "dst", "type"]
            }
        },
        {
            "name": "xerj_brain_unlink",
            "description": format!(
                "Retire a link in a second brain — never delete: the edge is \
                 soft-invalidated and stays queryable at past `as_of` moments. Get \
                 `edge_id` from ego responses. Idempotent: retiring an already-retired \
                 link reports `already_invalid_at`. Proxies \
                 DELETE /_graph/{{brain}}/link/{{edge_id}}.{BRAIN_HONESTY}"),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "brain": { "type": "string", "description": "Brain name." },
                    "edge_id": { "type": "string", "description": "The link's id, from an ego response." },
                    "invalid_at": { "description": "When the fact stopped being true: epoch-ms number or RFC3339 string (default: server now). The server separately records when it learned this (`expired_at`)." }
                },
                "required": ["brain", "edge_id"]
            }
        },
        {
            "name": "xerj_brain_overview",
            "description": format!(
                "Orientation call for a second brain: does it exist, live vs retired \
                 link counts, hub nodes (most linked), what taught it (per-detector \
                 counts), link types, and a created-over-time timeline. Cheap bounded \
                 aggregations, always current; pass `as_of` to summarize a past moment. \
                 Proxies GET /_graph/{{brain}}/overview.{BRAIN_HONESTY}"),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "brain": { "type": "string", "description": "Brain name." },
                    "as_of": { "description": "Bi-temporal cut: epoch-ms number or RFC3339 string (default now)." },
                    "top": { "type": "integer", "description": "Size of every top-N list; engine clamps to 1..=50 (default 10)." }
                },
                "required": ["brain"]
            }
        },
        {
            "name": "xerj_code_search",
            "description": concat!(
                "Reference-coding retrieval over a corpus of peer projects: how did ",
                "real codebases solve this problem? Returns the matching definition ",
                "(not a byte window), the file:line to cite, and per-hit licence ",
                "warnings — AGPL/SSPL/Elastic/BUSL/GPL/LGPL/MPL sources are approach-only. ",
                "Refuses indices older than 30 days (override with stale_ok). ",
                "A no-match is NOT an error: it says the corpus is wrong for the task. ",
                "Corpora: `xerj corpus list`."),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "corpus": { "type": "string", "description": "Corpus name (`xerj corpus list`)." },
                    "query": { "type": "string", "description": "What you need, in code terms (e.g. 'merge policy segment expiry')." },
                    "k": { "type": "integer", "description": "Passages to return; clamped to 1..=50 (default 5)." },
                    "mode": { "type": "string", "enum": ["bm25", "semantic", "hybrid"], "description": "Retrieval arm (default bm25 — measured best)." },
                    "lang": { "type": "string", "description": "Filter to one language field (e.g. rust, go)." },
                    "full": { "type": "integer", "description": "Max chars per passage (default 800; 0 = file head only)." },
                    "no_symbol": { "type": "boolean", "description": "Window selection instead of the matching definition." },
                    "stale_ok": { "type": "boolean", "description": "Override the 30-day staleness refusal." },
                    "licence_policy": { "type": "string", "enum": ["warn", "strict"], "description": "strict strips passage text from restricted-licence hits, keeping locator + warning (default warn)." }
                },
                "required": ["corpus", "query"]
            }
        }
    ])
}

/// Dispatch a `tools/call` to the matching proxy. Any bad-argument or transport
/// problem is returned as an `isError:true` tool result (not a protocol error),
/// which is the MCP convention for tool-execution failures.
async fn call_tool(ctx: &Ctx, msg: &Value) -> Value {
    let params = match msg.get("params") {
        Some(p) => p,
        None => return tool_text("missing `params`", true),
    };
    let name = match params.get("name").and_then(Value::as_str) {
        Some(n) => n,
        None => return tool_text("missing tool `name`", true),
    };
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let result = match name {
        "xerj_search" => build_search(&args),
        "xerj_semantic_search" => build_semantic(&args),
        "xerj_vector_search" => build_vector(&args),
        "xerj_hybrid_search" => build_hybrid(&args),
        "xerj_memory_store" => build_memory_store(&args),
        "xerj_memory_recall" => build_memory_recall(&args),
        "xerj_brain_ego" => build_brain_ego(&args),
        "xerj_brain_link" => build_brain_link(&args),
        "xerj_brain_unlink" => build_brain_unlink(&args),
        "xerj_brain_overview" => build_brain_overview(&args),
        "xerj_code_search" => return run_code_search(ctx, &args).await,
        other => return tool_text(format!("unknown tool: {other}"), true),
    };

    match result {
        Ok((method, path, body)) => engine_request(ctx, method, &path, body).await,
        Err(msg) => tool_text(msg, true),
    }
}

/// HTTP method of a proxied engine request. The engine's graph surface is not
/// POST-only (ego/overview are GET, unlink is DELETE), so builders carry it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Get,
    Post,
    Delete,
}

/// What every `build_*` function returns: the request to proxy, fully formed
/// (query params already encoded into the path) — pure and unit-testable.
type BuiltRequest = (Method, String, Option<Value>);

// ── Argument helpers ────────────────────────────────────────────────────────

fn req_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing or empty required string field `{key}`"))
}

/// Copy an optional field from `args` into `body` under the same key, if present.
fn copy_opt(args: &Value, body: &mut serde_json::Map<String, Value>, key: &str) {
    if let Some(v) = args.get(key) {
        if !v.is_null() {
            body.insert(key.to_string(), v.clone());
        }
    }
}

/// Percent-encode one path segment or query value (RFC 3986 unreserved set
/// passes through; everything else, including `/`, is `%XX`-encoded). Node
/// ids are arbitrary document ids, so this is not optional.
fn enc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Join `(key, value)` pairs into a query string, percent-encoding values
/// (keys are fixed identifiers). Empty input → empty string (no `?`).
fn qs(pairs: &[(&str, String)]) -> String {
    if pairs.is_empty() {
        return String::new();
    }
    let joined: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{k}={}", enc(v)))
        .collect();
    format!("?{}", joined.join("&"))
}

/// Read an optional bi-temporal instant argument: an epoch-ms number or an
/// RFC3339 string, stringified for the query line (the engine parses both).
fn opt_instant(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(Some(n.to_string())),
        Some(Value::String(s)) if !s.is_empty() => Ok(Some(s.clone())),
        Some(_) => Err(format!(
            "`{key}` must be an epoch-ms number or an RFC3339 string"
        )),
    }
}

/// Read an optional argument of one exact JSON type. Present-but-mistyped is a
/// builder error, never a silent drop: an agent that sends
/// `include_expired: "true"` must be told, not handed the default and left
/// believing it saw history. (Same policy `opt_instant` already applies.)
fn opt_typed<'a, T>(
    args: &'a Value,
    key: &str,
    want: &str,
    read: impl Fn(&'a Value) -> Option<T>,
) -> Result<Option<T>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => read(v)
            .map(Some)
            .ok_or_else(|| format!("`{key}` must be {want}")),
    }
}

// ── Per-tool request builders → (method, path, body) ────────────────────────

/// Normalise the optional `rerank` argument into the `_search` body.
///
/// `plain_query` is the tool's `query` when the caller passed a plain string.
/// It matters because the string path expands into the definition-first `bool`
/// shape, and the node refuses to read a question out of a `bool` tree — so
/// without this every plain-string search that asked for reranking would come
/// back 400. The caller's words ARE the question; they are carried across.
///
/// Refusals here mirror the node's, using the node's own validator
/// ([`xerj_rerank::RerankConfig::from_json`]), so an agent is told about a
/// misspelt key without paying for a round trip. Present-but-mistyped is an
/// error, never a silent drop (the `opt_typed` policy): an agent that asked for
/// reranking and silently did not get it would trust an order nobody judged.
fn apply_rerank(
    args: &Value,
    body: &mut serde_json::Map<String, Value>,
    plain_query: Option<&str>,
    query_required: bool,
) -> Result<(), String> {
    let mut block = match args.get("rerank") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => return Ok(()),
        Some(Value::Bool(true)) => serde_json::Map::new(),
        Some(Value::Object(o)) => o.clone(),
        Some(_) => {
            return Err(
                "`rerank` must be true, or an object such as {\"min_score\": 0.5} \
                 (pass {} for defaults)"
                    .to_string(),
            )
        }
    };
    if body.contains_key("sort") {
        return Err(
            "`rerank` cannot be combined with `sort`: an explicit sort already fixes the \
             order, and reranking would discard it. Drop one of them"
                .to_string(),
        );
    }
    if !block.contains_key("query") {
        match plain_query {
            Some(q) => {
                block.insert("query".into(), json!(q));
            }
            None if query_required => {
                return Err(
                    "`rerank.query` is required for a hybrid search: it fuses several \
                     sub-queries, so there is no single question to judge documents \
                     against. Pass the natural-language question, e.g. \
                     {\"rerank\": {\"query\": \"why did the vpn drop?\"}}"
                        .to_string(),
                )
            }
            // A DSL object: the node reads the question from `match`,
            // `match_phrase`, `multi_match`, `semantic` and `simple_query_string`,
            // and refuses by name for anything else. Left to it, so there is
            // one implementation of that rule.
            None => {}
        }
    }
    let block = Value::Object(block);
    xerj_rerank::RerankConfig::from_json(&block).map_err(|e| e.to_string())?;
    body.insert("rerank".into(), block);
    Ok(())
}

fn build_search(args: &Value) -> Result<BuiltRequest, String> {
    // #962: `index` used to be required, but nothing an agent reads said what
    // to pass — the first real tool call after `xerj autoindex` was an error
    // until the agent guessed `ax-*`. Default to the prefix autoindex writes
    // (matching the CLI's `xerj search`, which defaults to "ax"); an explicit
    // index still wins.
    let index = args
        .get("index")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("ax-*");
    let mut body = serde_json::Map::new();
    let mut plain_string = false;
    match args.get("query") {
        // A plain string gets the definition-first shape `xerj search` uses
        // (measured 38%→94% correct-file-at-rank-1 on cross-file lookups vs a
        // bare body match). Agents should never have to hand-write ES DSL for
        // the common "find me this code" case; the DSL object path stays
        // untouched for structured queries.
        Some(Value::String(q)) if !q.trim().is_empty() => {
            plain_string = true;
            body.insert(
                "query".into(),
                json!({ "bool": { "should": [
                    { "multi_match": {
                        "query": q,
                        "fields": ["name^4", "defs^3", "code^2", "body", "title"],
                        "type": "most_fields"
                    }},
                    { "match_phrase": { "defs": { "query": q, "boost": 8 } } },
                    { "term": { "name": { "value": q, "boost": 10 } } }
                ]}}),
            );
        }
        Some(q) if !q.is_null() => {
            body.insert("query".into(), q.clone());
        }
        _ => {
            body.insert("query".into(), json!({ "match_all": {} }));
        }
    }
    for k in ["size", "from", "sort", "_source"] {
        copy_opt(args, &mut body, k);
    }
    let plain_query = match args.get("query") {
        Some(Value::String(q)) if plain_string => Some(q.as_str()),
        _ => None,
    };
    apply_rerank(args, &mut body, plain_query, false)?;
    // String-mode defaults, only where the caller left them unset: project
    // `_source` down to the citation fields and ask for `_passage` so the
    // response is the matching snippet, not the whole file body (measured ~9x
    // smaller for 3 code hits).
    if plain_string {
        body.entry("_source".to_string()).or_insert_with(|| {
            json!([
                "ax_path",
                "path",
                "line",
                "start_line",
                "language",
                "kind",
                "code",
                "title"
            ])
        });
        body.entry("fields".to_string())
            .or_insert_with(|| json!(["_passage"]));
    }
    Ok((
        Method::Post,
        format!("/{index}/_search"),
        Some(Value::Object(body)),
    ))
}

fn build_semantic(args: &Value) -> Result<BuiltRequest, String> {
    let index = req_str(args, "index")?;
    let field = req_str(args, "field")?;
    let query = req_str(args, "query")?;
    let k = args.get("k").and_then(Value::as_u64).unwrap_or(10);

    let mut semantic = json!({ "field": field, "query": query, "k": k });
    if let Some(f) = args.get("filter") {
        if !f.is_null() {
            semantic["filter"] = f.clone();
        }
    }
    // Also cap `size` at k so the response isn't padded past the requested set.
    let body = json!({ "query": { "semantic": semantic }, "size": k });
    Ok((Method::Post, format!("/{index}/_search"), Some(body)))
}

fn build_vector(args: &Value) -> Result<BuiltRequest, String> {
    let index = req_str(args, "index")?;
    let field = req_str(args, "field")?;
    let vector = args
        .get("query_vector")
        .filter(|v| v.is_array())
        .ok_or_else(|| "missing required array field `query_vector`".to_string())?;
    let k = args.get("k").and_then(Value::as_u64).unwrap_or(10);

    let mut knn = serde_json::Map::new();
    knn.insert("field".into(), json!(field));
    knn.insert("query_vector".into(), vector.clone());
    knn.insert("k".into(), json!(k));
    if let Some(nc) = args.get("num_candidates").and_then(Value::as_u64) {
        knn.insert("num_candidates".into(), json!(nc));
    }
    if let Some(f) = args.get("filter") {
        if !f.is_null() {
            knn.insert("filter".into(), f.clone());
        }
    }
    let body = json!({ "knn": Value::Object(knn), "size": k });
    Ok((Method::Post, format!("/{index}/_search"), Some(body)))
}

fn build_hybrid(args: &Value) -> Result<BuiltRequest, String> {
    let index = req_str(args, "index")?;
    let queries = args
        .get("queries")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| "missing or empty required array field `queries`".to_string())?;

    let mut hybrid = serde_json::Map::new();
    hybrid.insert("queries".into(), Value::Array(queries.clone()));
    if let Some(f) = args.get("fusion") {
        if !f.is_null() {
            hybrid.insert("fusion".into(), f.clone());
        }
    }
    let mut body = serde_json::Map::new();
    body.insert("query".into(), json!({ "hybrid": Value::Object(hybrid) }));
    copy_opt(args, &mut body, "size");
    apply_rerank(args, &mut body, None, true)?;
    Ok((
        Method::Post,
        format!("/{index}/_search"),
        Some(Value::Object(body)),
    ))
}

fn build_memory_store(args: &Value) -> Result<BuiltRequest, String> {
    let namespace = req_str(args, "namespace")?;
    // `text` is required unless a raw `vector` is supplied; the engine enforces
    // "non-empty text or a vector", so mirror that leniently here.
    let has_text = args
        .get("text")
        .and_then(Value::as_str)
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let has_vector = args.get("vector").map(Value::is_array).unwrap_or(false);
    if !has_text && !has_vector {
        return Err("a memory must have non-empty `text` or a `vector`".to_string());
    }

    let mut body = serde_json::Map::new();
    for k in [
        "text",
        "metadata",
        "id",
        "vector",
        "dedup",
        "dedup_threshold",
    ] {
        copy_opt(args, &mut body, k);
    }
    Ok((
        Method::Post,
        format!("/_memory/{namespace}"),
        Some(Value::Object(body)),
    ))
}

fn build_memory_recall(args: &Value) -> Result<BuiltRequest, String> {
    let namespace = req_str(args, "namespace")?;
    let mut body = serde_json::Map::new();
    for k in [
        "query",
        "vector",
        "semantic",
        "hybrid",
        "fusion",
        "k",
        "filter",
        "recency_weight",
        "graph",
    ] {
        copy_opt(args, &mut body, k);
    }
    Ok((
        Method::Post,
        format!("/_memory/{namespace}/_recall"),
        Some(Value::Object(body)),
    ))
}

// ── Second-brain builders (graph surface: GET/POST/DELETE) ──────────────────

fn build_brain_ego(args: &Value) -> Result<BuiltRequest, String> {
    let brain = req_str(args, "brain")?;
    let node = req_str(args, "node")?;

    let mut q: Vec<(&str, String)> = vec![("node", node.to_string())];
    if let Some(h) = opt_typed(args, "hops", "an integer (1 or 2)", Value::as_u64)? {
        q.push(("hops", h.to_string()));
    }
    if let Some(d) = opt_typed(args, "direction", "a string (out|in|both)", Value::as_str)? {
        q.push(("direction", d.to_string()));
    }
    // The HTTP surface takes a comma-separated list; the schema takes an
    // array (better for models). Join here.
    if let Some(types) = opt_typed(args, "types", "an array of strings", |v| {
        v.as_array()
            .and_then(|a| a.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
    })? {
        if !types.is_empty() {
            q.push(("types", types.join(",")));
        }
    }
    if let Some(l) = opt_typed(args, "limit", "an integer", Value::as_u64)? {
        q.push(("limit", l.to_string()));
    }
    if let Some(a) = opt_instant(args, "as_of")? {
        q.push(("as_of", a));
    }
    if let Some(b) = opt_typed(args, "include_expired", "a boolean", Value::as_bool)? {
        q.push(("include_expired", b.to_string()));
    }
    // Deliberate default flip vs the HTTP surface: agents nearly always want
    // node previews, so the proxy sets include_nodes explicitly, default true.
    let include_nodes =
        opt_typed(args, "include_nodes", "a boolean", Value::as_bool)?.unwrap_or(true);
    q.push(("include_nodes", include_nodes.to_string()));

    Ok((
        Method::Get,
        format!("/_graph/{}/ego{}", enc(brain), qs(&q)),
        None,
    ))
}

fn build_brain_link(args: &Value) -> Result<BuiltRequest, String> {
    let brain = req_str(args, "brain")?;
    let src = req_str(args, "src")?;
    let dst = req_str(args, "dst")?;
    let edge_type = req_str(args, "type")?;

    let mut body = serde_json::Map::new();
    body.insert("src".into(), json!(src));
    body.insert("dst".into(), json!(dst));
    body.insert("type".into(), json!(edge_type));
    for k in ["evidence", "weight", "confidence", "valid_at"] {
        copy_opt(args, &mut body, k);
    }
    Ok((
        Method::Post,
        format!("/_graph/{}/link", enc(brain)),
        Some(Value::Object(body)),
    ))
}

fn build_brain_unlink(args: &Value) -> Result<BuiltRequest, String> {
    let brain = req_str(args, "brain")?;
    let edge_id = req_str(args, "edge_id")?;
    let mut q: Vec<(&str, String)> = Vec::new();
    if let Some(at) = opt_instant(args, "invalid_at")? {
        q.push(("invalid_at", at));
    }
    Ok((
        Method::Delete,
        format!("/_graph/{}/link/{}{}", enc(brain), enc(edge_id), qs(&q)),
        None,
    ))
}

fn build_brain_overview(args: &Value) -> Result<BuiltRequest, String> {
    let brain = req_str(args, "brain")?;
    let mut q: Vec<(&str, String)> = Vec::new();
    if let Some(a) = opt_instant(args, "as_of")? {
        q.push(("as_of", a));
    }
    if let Some(t) = opt_typed(args, "top", "an integer", Value::as_u64)? {
        q.push(("top", t.to_string()));
    }
    Ok((
        Method::Get,
        format!("/_graph/{}/overview{}", enc(brain), qs(&q)),
        None,
    ))
}

// ── xerj_code_search: the shared reference-coding pipeline ──────────────────

/// Synchronous [`XcHttp`] over a fresh reqwest client. This crate is
/// deliberately async-only (no `blocking` feature), so each call runs on a
/// private current-thread runtime inside the `spawn_blocking` below — no
/// query logic lives here, only transport.
struct McpXc {
    base_url: String,
    auth: Option<String>,
}

impl McpXc {
    fn request(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("runtime: {e}"))?;
        let url = format!("{}{}", self.base_url, path);
        let auth = self.auth.clone();
        let method = method.to_string();
        let body = body.cloned();
        rt.block_on(async move {
            let client = reqwest::Client::new();
            let m = reqwest::Method::from_bytes(method.as_bytes())
                .map_err(|_| format!("not an HTTP method: {method}"))?;
            let mut req = client.request(m, &url);
            if let Some(h) = auth.as_deref() {
                req = req.header("Authorization", h);
            }
            if let Some(b) = &body {
                req = req.json(b);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| format!("no response from {url}: {e}"))?;
            // Surface the status WITH the body: the tri-state callers (404 =
            // zero, not error) classify it themselves.
            let status = resp.status().as_u16();
            let text = resp.text().await.map_err(|e| format!("read {url}: {e}"))?;
            let v: Value = serde_json::from_str(&text).unwrap_or(Value::String(text));
            match status {
                200..=299 => Ok(v),
                404 => Err("__404__".to_string()),
                other => Err(format!("HTTP {other}: {v}")),
            }
        })
    }
}

impl xerj_common::xccode::XcHttp for McpXc {
    fn get_mapping(&self, path: &str) -> Result<Value, String> {
        self.request("GET", path, None)
    }
    fn cat_indices_json(&self, pattern: &str) -> Result<Vec<String>, String> {
        let v = match self.request(
            "GET",
            &format!("/_cat/indices/{pattern}?format=json&h=index"),
            None,
        ) {
            Ok(v) => v,
            // A 404 on a wildcard means "no such indices" -> none, not an error.
            Err(e) if e.contains("__404__") => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        v.as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| r.get("index").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect()
            })
            .ok_or_else(|| "non-array _cat reply".to_string())
    }
    fn search(&self, index: &str, body: &Value) -> Result<Value, String> {
        self.request("POST", &format!("/{index}/_search"), Some(body))
    }
}

/// `xerj_code_search` — the SAME pipeline `xerj code` runs, from the SAME
/// renderer; only the surface differs. Warnings ride the TOP of the tool text
/// (MCP has no stderr channel, so warnings are content), and the exit-code
/// triangle maps onto `isError`: staleness/not-loaded/transport are errors, a
/// no-match is NOT.
async fn run_code_search(ctx: &Ctx, args: &Value) -> Value {
    let corpus = match args.get("corpus").and_then(Value::as_str) {
        Some(c) if !c.is_empty() => c.to_string(),
        _ => return tool_text("xerj_code_search: `corpus` is required", true),
    };
    let query = match args.get("query").and_then(Value::as_str) {
        Some(q) if !q.is_empty() => q.to_string(),
        _ => return tool_text("xerj_code_search: `query` is required", true),
    };
    let mut p = xerj_common::xccode::CodeParams::new(&corpus, &query);
    if let Some(k) = args.get("k") {
        match k.as_u64().filter(|n| (1..=50).contains(n)) {
            Some(n) => p.k = n as usize,
            None => return tool_text("xerj_code_search: `k` must be an integer 1..=50", true),
        }
    }
    if let Some(m) = args.get("mode").and_then(Value::as_str) {
        match xerj_common::xccode::Mode::parse(m) {
            Some(mode) => p.mode = mode,
            None => return tool_text(format!("xerj_code_search: unknown mode '{m}'"), true),
        }
    }
    if let Some(l) = args.get("lang").and_then(Value::as_str) {
        p.lang = Some(l.to_string());
    }
    match args.get("full").and_then(Value::as_u64) {
        Some(n) => p.full = n as usize,
        None => {
            if args.get("full").is_some() {
                return tool_text(
                    "xerj_code_search: `full` must be a non-negative integer",
                    true,
                );
            }
        }
    }
    p.no_symbol = args
        .get("no_symbol")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    p.stale_ok = args
        .get("stale_ok")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    p.strict_licence = matches!(
        args.get("licence_policy").and_then(Value::as_str),
        Some("strict")
    );

    let base_url = ctx.base_url.clone();
    let auth = ctx.auth.clone();
    let res = tokio::task::spawn_blocking(move || {
        let http = McpXc { base_url, auth };
        let root = code_root();
        xerj_common::xccode::run_code_query(&root, &http, &http.base_url, &p, "stale_ok:true")
    })
    .await;

    let out = match res {
        Ok(out) => out,
        Err(e) => return tool_text(format!("xerj_code_search: join error: {e}"), true),
    };
    let mut text = String::new();
    for w in &out.warnings {
        text.push_str(w);
        text.push('\n');
    }
    text.push_str(&out.text);
    tool_text(text, out.is_error)
}

/// `~/.xerj-code` unless `XERJ_CODE_HOME` says otherwise — the SAME root the
/// CLI uses, so both surfaces see the same corpora.
fn code_root() -> std::path::PathBuf {
    if let Ok(home) = std::env::var("XERJ_CODE_HOME") {
        if !home.is_empty() {
            return std::path::PathBuf::from(home);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::Path::new(&home).join(".xerj-code")
}

// ── Engine transport ─────────────────────────────────────────────────────────

/// Send one request to the configured XERJ base URL and wrap the response as
/// an MCP tool result. Non-2xx responses (and transport errors) come back as
/// `isError:true` with the engine's text verbatim, so the agent sees exactly
/// what the engine said (including its "not a graph database" refusals).
async fn engine_request(ctx: &Ctx, method: Method, path: &str, body: Option<Value>) -> Value {
    let url = format!("{}{}", ctx.base_url, path);
    let mut req = match method {
        Method::Get => ctx.client.get(&url),
        Method::Post => ctx.client.post(&url),
        Method::Delete => ctx.client.delete(&url),
    };
    if let Some(body) = &body {
        req = req.json(body);
    }
    if let Some(auth) = &ctx.auth {
        req = req.header("Authorization", auth);
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            if status.is_success() {
                tool_text(text, false)
            } else {
                tool_text(format!("XERJ returned HTTP {status}: {text}"), true)
            }
        }
        Err(e) => tool_text(format!("request to {url} failed: {e}"), true),
    }
}

// ─────────────────────────────── Tests ─────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Unwrap a builder result into `(method, path, body)` with a non-optional
    /// body, for the POST builders that always carry one.
    fn built(r: Result<BuiltRequest, String>) -> (Method, String, Value) {
        let (m, p, b) = r.unwrap();
        (m, p, b.expect("builder should carry a body"))
    }

    #[test]
    fn search_defaults_to_match_all() {
        let (method, path, body) = built(build_search(&json!({ "index": "docs" })));
        assert_eq!(method, Method::Post);
        assert_eq!(path, "/docs/_search");
        assert_eq!(body["query"], json!({ "match_all": {} }));
    }

    #[test]
    fn search_passes_query_and_size() {
        let (_, _, body) = built(build_search(&json!({
            "index": "docs",
            "query": { "match": { "body": "rust" } },
            "size": 5
        })));
        assert_eq!(body["query"]["match"]["body"], "rust");
        assert_eq!(body["size"], 5);
    }

    #[test]
    fn search_index_defaults_to_the_autoindex_prefix() {
        // #962: an agent that ran `xerj autoindex` and then called
        // xerj_search with no index got isError on its first real tool
        // call — nothing said the corpus lives under `ax-*`. The default
        // now matches what autoindex writes (and what `xerj search` uses).
        let (_, path, _) = built(build_search(&json!({})));
        assert_eq!(path, "/ax-*/_search");
        // Whitespace-only is treated as absent; explicit index still wins.
        let (_, path, _) = built(build_search(&json!({ "index": "  " })));
        assert_eq!(path, "/ax-*/_search");
        let (_, path, _) = built(build_search(&json!({ "index": "docs" })));
        assert_eq!(path, "/docs/_search");
    }

    #[test]
    fn semantic_builds_semantic_query_and_caps_size() {
        let (method, path, body) = built(build_semantic(&json!({
            "index": "kb", "field": "content", "query": "how to reset", "k": 3
        })));
        assert_eq!(method, Method::Post);
        assert_eq!(path, "/kb/_search");
        assert_eq!(body["query"]["semantic"]["field"], "content");
        assert_eq!(body["query"]["semantic"]["query"], "how to reset");
        assert_eq!(body["query"]["semantic"]["k"], 3);
        assert_eq!(body["size"], 3);
    }

    #[test]
    fn vector_builds_top_level_knn() {
        let (_, path, body) = built(build_vector(&json!({
            "index": "emb", "field": "vec", "query_vector": [0.1, 0.2, 0.3], "k": 4
        })));
        assert_eq!(path, "/emb/_search");
        assert_eq!(body["knn"]["field"], "vec");
        assert_eq!(body["knn"]["query_vector"], json!([0.1, 0.2, 0.3]));
        assert_eq!(body["knn"]["k"], 4);
        assert_eq!(body["size"], 4);
    }

    #[test]
    fn vector_requires_query_vector() {
        assert!(build_vector(&json!({ "index": "e", "field": "v" })).is_err());
    }

    #[test]
    fn hybrid_wraps_queries_and_fusion() {
        let (_, path, body) = built(build_hybrid(&json!({
            "index": "h",
            "queries": [
                { "query": { "match": { "body": "cats" } }, "weight": 1.0 },
                { "query": { "knn": { "field": "v", "query_vector": [0.1], "k": 5 } }, "weight": 0.3 }
            ],
            "fusion": "rrf",
            "size": 10
        })));
        assert_eq!(path, "/h/_search");
        assert_eq!(body["query"]["hybrid"]["fusion"], "rrf");
        assert_eq!(
            body["query"]["hybrid"]["queries"].as_array().unwrap().len(),
            2
        );
        assert_eq!(body["size"], 10);
    }

    // ── `rerank` argument ───────────────────────────────────────────────────

    #[test]
    fn rerank_is_absent_unless_asked_for() {
        for args in [
            json!({ "index": "d", "query": "how do refunds work" }),
            json!({ "index": "d", "query": "how do refunds work", "rerank": null }),
            json!({ "index": "d", "query": "how do refunds work", "rerank": false }),
        ] {
            let (_, _, body) = built(build_search(&args));
            assert!(
                body.get("rerank").is_none(),
                "no `rerank` block means no provider call and no egress: {body}"
            );
        }
    }

    #[test]
    fn a_plain_string_query_becomes_the_rerank_question() {
        // The string path expands to a `bool`, which the node refuses to read a
        // question from. Without carrying the string across, every plain-string
        // search that asked for reranking would be a 400.
        let (_, _, body) = built(build_search(&json!({
            "index": "d", "query": "how do refunds work", "rerank": true
        })));
        assert!(body["query"]["bool"].is_object(), "still definition-first");
        assert_eq!(body["rerank"]["query"], "how do refunds work");

        let (_, _, body) = built(build_search(&json!({
            "index": "d", "query": "how do refunds work", "rerank": { "min_score": 0.5 }
        })));
        assert_eq!(body["rerank"]["query"], "how do refunds work");
        assert_eq!(body["rerank"]["min_score"], 0.5);
    }

    #[test]
    fn an_explicit_rerank_query_is_never_overwritten() {
        let (_, _, body) = built(build_search(&json!({
            "index": "d", "query": "refund",
            "rerank": { "query": "What is the refund window for annual plans?" }
        })));
        assert_eq!(
            body["rerank"]["query"],
            "What is the refund window for annual plans?"
        );
    }

    #[test]
    fn a_dsl_query_leaves_question_inference_to_the_node() {
        let (_, _, body) = built(build_search(&json!({
            "index": "d", "query": { "match": { "body": "refund window" } }, "rerank": {}
        })));
        assert_eq!(
            body["rerank"],
            json!({}),
            "one implementation of inference: the node's"
        );
    }

    #[test]
    fn rerank_refusals_happen_before_the_round_trip() {
        let bad = |args: Value| build_search(&args).expect_err("must be refused");

        let e = bad(json!({ "index": "d", "query": "q", "rerank": "yes" }));
        assert!(e.contains("must be true, or an object"), "{e}");

        let e = bad(json!({ "index": "d", "query": "q", "sort": ["_doc"], "rerank": true }));
        assert!(e.contains("sort"), "{e}");

        // The node's own validator: a misspelt key is named, not dropped.
        let e = bad(json!({ "index": "d", "query": "q", "rerank": { "treshold": 0.5 } }));
        assert!(e.contains("treshold"), "{e}");

        let e = bad(json!({ "index": "d", "query": "q", "rerank": { "min_score": 7 } }));
        assert!(e.contains("probability"), "{e}");

        let e = bad(json!({ "index": "d", "query": "q", "rerank": { "window": 100000 } }));
        assert!(e.contains("rerank.window"), "{e}");
    }

    #[test]
    fn hybrid_rerank_requires_the_question() {
        let queries = json!([{ "query": { "match": { "body": "vpn" } } }]);
        let e = build_hybrid(&json!({ "index": "h", "queries": queries, "rerank": true }))
            .expect_err("a hybrid has no single question");
        assert!(e.contains("rerank.query"), "{e}");

        let (_, path, body) = built(build_hybrid(&json!({
            "index": "h", "queries": queries,
            "rerank": { "query": "why did the vpn drop?", "window": 20 }
        })));
        assert_eq!(path, "/h/_search");
        assert_eq!(body["rerank"]["query"], "why did the vpn drop?");
        assert_eq!(body["rerank"]["window"], 20);
        assert!(body["query"]["hybrid"].is_object());
    }

    #[test]
    fn the_rerank_description_tells_an_agent_what_it_needs_to_decide() {
        let specs = tool_specs();
        for tool in ["xerj_search", "xerj_hybrid_search"] {
            let spec = specs
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == tool)
                .unwrap();
            let d = spec["inputSchema"]["properties"]["rerank"]["description"]
                .as_str()
                .unwrap_or_else(|| panic!("{tool} must describe `rerank`"));
            for needle in [
                "WORTH IT WHEN",
                "NOT WORTH IT FOR",
                "NEEDS A CONFIGURED PROVIDER",
                "503",
                // The honest egress statement: rerank is the only SEARCH-TIME
                // path, and the description names the other two that send text
                // so an agent never repeats "nothing else leaves the machine"
                // on a node running proxy embeddings or a WAL tap.
                "only search-time feature that sends document text off the node",
                "default_endpoint",
                "WAL tap",
                "never tell a user that nothing else leaves the machine",
                "_rerank",
            ] {
                assert!(d.contains(needle), "{tool}: description lacks `{needle}`");
            }
            // The first corrected wording ("two other ... paths") was incomplete: the neural model
            // download and cluster Raft traffic are outbound too (they carry no
            // document text). docs/RERANK.md holds the complete list.
            assert!(
                !d.to_lowercase().contains("other outbound paths"),
                "{tool}: {d}"
            );
            // The quoted ceilings are the node's, not a copy of them.
            assert!(
                d.contains(&format!("max {}", xerj_rerank::MAX_WINDOW)),
                "{d}"
            );
            assert!(
                !spec["inputSchema"]["required"]
                    .as_array()
                    .map(|r| r.iter().any(|r| r == "rerank"))
                    .unwrap_or(false),
                "{tool}: `rerank` is optional"
            );
        }
    }

    #[test]
    fn memory_store_requires_text_or_vector() {
        assert!(build_memory_store(&json!({ "namespace": "n" })).is_err());
        let (_, path, body) = built(build_memory_store(
            &json!({ "namespace": "n", "text": "remember this" }),
        ));
        assert_eq!(path, "/_memory/n");
        assert_eq!(body["text"], "remember this");
    }

    #[test]
    fn memory_recall_path_and_passthrough() {
        let (_, path, body) = built(build_memory_recall(&json!({
            "namespace": "n", "query": "what did I say", "semantic": true, "k": 5
        })));
        assert_eq!(path, "/_memory/n/_recall");
        assert_eq!(body["semantic"], true);
        assert_eq!(body["k"], 5);
    }

    #[test]
    fn memory_recall_passes_graph_coupling_verbatim() {
        let graph = json!({
            "mode": "restrict",
            "seeds": ["note-a"],
            "hops": 2,
            "as_of": 1750000000000_i64
        });
        let (_, path, body) = built(build_memory_recall(&json!({
            "namespace": "n", "query": "deploy steps", "graph": graph
        })));
        assert_eq!(path, "/_memory/n/_recall");
        assert_eq!(body["graph"], graph);
        // Absent → key absent (graph-less recall stays bit-identical).
        let (_, _, body) = built(build_memory_recall(
            &json!({ "namespace": "n", "query": "x" }),
        ));
        assert!(body.get("graph").is_none());
    }

    // ── second-brain builders ───────────────────────────────────────────────

    #[test]
    fn brain_ego_minimal_sets_include_nodes_true() {
        let (method, path, body) =
            build_brain_ego(&json!({ "brain": "notes", "node": "note-a" })).unwrap();
        assert_eq!(method, Method::Get);
        assert_eq!(path, "/_graph/notes/ego?node=note-a&include_nodes=true");
        assert!(body.is_none(), "GET must carry no body");
    }

    #[test]
    fn brain_ego_include_nodes_false_is_respected() {
        let (_, path, _) = build_brain_ego(&json!({
            "brain": "notes", "node": "note-a", "include_nodes": false
        }))
        .unwrap();
        assert_eq!(path, "/_graph/notes/ego?node=note-a&include_nodes=false");
    }

    #[test]
    fn brain_ego_encodes_all_params() {
        let (_, path, _) = build_brain_ego(&json!({
            "brain": "notes",
            "node": "dir/some note.md",
            "hops": 2,
            "direction": "out",
            "types": ["references", "mentions"],
            "limit": 50,
            "as_of": 1750000000000_i64,
            "include_expired": true
        }))
        .unwrap();
        assert_eq!(
            path,
            "/_graph/notes/ego?node=dir%2Fsome%20note.md&hops=2&direction=out\
             &types=references%2Cmentions&limit=50&as_of=1750000000000\
             &include_expired=true&include_nodes=true"
        );
    }

    #[test]
    fn brain_ego_accepts_rfc3339_as_of() {
        let (_, path, _) = build_brain_ego(&json!({
            "brain": "b", "node": "n", "as_of": "2026-05-02T00:00:00Z"
        }))
        .unwrap();
        assert!(path.contains("as_of=2026-05-02T00%3A00%3A00Z"));
        // Wrong type is a builder-side error, not a silent drop.
        assert!(build_brain_ego(&json!({ "brain": "b", "node": "n", "as_of": true })).is_err());
    }

    #[test]
    fn brain_ego_requires_brain_and_node() {
        assert!(build_brain_ego(&json!({ "node": "n" })).is_err());
        assert!(build_brain_ego(&json!({ "brain": "b" })).is_err());
    }

    #[test]
    fn brain_link_builds_post_body() {
        let (method, path, body) = built(build_brain_link(&json!({
            "brain": "notes",
            "src": "note-a",
            "dst": "note-b",
            "type": "references",
            "evidence": { "quote": "see [[note-b]]", "source": "note-a", "offset": 120 },
            "confidence": 0.8,
            "valid_at": 1750000000000_i64
        })));
        assert_eq!(method, Method::Post);
        assert_eq!(path, "/_graph/notes/link");
        assert_eq!(body["src"], "note-a");
        assert_eq!(body["dst"], "note-b");
        assert_eq!(body["type"], "references");
        assert_eq!(body["evidence"]["quote"], "see [[note-b]]");
        assert_eq!(body["confidence"], 0.8);
        assert_eq!(body["valid_at"], 1750000000000_i64);
    }

    #[test]
    fn brain_link_requires_src_dst_type() {
        let base = json!({ "brain": "b", "src": "a", "dst": "c", "type": "t" });
        assert!(build_brain_link(&base).is_ok());
        for missing in ["brain", "src", "dst", "type"] {
            let mut args = base.clone();
            args.as_object_mut().unwrap().remove(missing);
            assert!(build_brain_link(&args).is_err(), "should require {missing}");
        }
    }

    #[test]
    fn brain_unlink_builds_delete_with_invalid_at() {
        let (method, path, body) = build_brain_unlink(&json!({
            "brain": "notes", "edge_id": "bef814a75bd3d914c3e561f610154304"
        }))
        .unwrap();
        assert_eq!(method, Method::Delete);
        assert_eq!(path, "/_graph/notes/link/bef814a75bd3d914c3e561f610154304");
        assert!(body.is_none());

        let (_, path, _) = build_brain_unlink(&json!({
            "brain": "notes",
            "edge_id": "bef814a75bd3d914c3e561f610154304",
            "invalid_at": 1750000000000_i64
        }))
        .unwrap();
        assert_eq!(
            path,
            "/_graph/notes/link/bef814a75bd3d914c3e561f610154304?invalid_at=1750000000000"
        );
    }

    #[test]
    fn brain_unlink_requires_edge_id() {
        assert!(build_brain_unlink(&json!({ "brain": "b" })).is_err());
    }

    #[test]
    fn brain_overview_builds_get_with_params() {
        let (method, path, body) = build_brain_overview(&json!({ "brain": "notes" })).unwrap();
        assert_eq!(method, Method::Get);
        assert_eq!(path, "/_graph/notes/overview");
        assert!(body.is_none());

        let (_, path, _) = build_brain_overview(&json!({
            "brain": "notes", "as_of": "2026-05-02T00:00:00Z", "top": 5
        }))
        .unwrap();
        assert_eq!(
            path,
            "/_graph/notes/overview?as_of=2026-05-02T00%3A00%3A00Z&top=5"
        );
    }

    #[test]
    fn brain_builders_reject_mistyped_args_never_drop() {
        // Each of these was once silently dropped (defaults applied) — an
        // honesty bug: the agent believed it filtered/replayed but did not.
        let cases = [
            json!({ "brain": "b", "node": "n", "hops": "2" }),
            json!({ "brain": "b", "node": "n", "direction": 5 }),
            json!({ "brain": "b", "node": "n", "types": "references" }),
            json!({ "brain": "b", "node": "n", "types": [1] }),
            json!({ "brain": "b", "node": "n", "limit": "50" }),
            json!({ "brain": "b", "node": "n", "include_expired": "true" }),
            json!({ "brain": "b", "node": "n", "include_nodes": "false" }),
        ];
        for args in &cases {
            assert!(build_brain_ego(args).is_err(), "should reject {args}");
        }
        assert!(build_brain_overview(&json!({ "brain": "b", "top": "5" })).is_err());
        // Explicit null still means "absent", not an error.
        assert!(build_brain_ego(&json!({ "brain": "b", "node": "n", "hops": null })).is_ok());
    }

    #[test]
    fn enc_percent_encodes_reserved_bytes() {
        assert_eq!(enc("plain-id_0.9~x"), "plain-id_0.9~x");
        assert_eq!(enc("a/b c?d&e=f"), "a%2Fb%20c%3Fd%26e%3Df");
    }

    #[test]
    fn initialize_echoes_client_protocol_version() {
        let msg = json!({ "params": { "protocolVersion": "2024-11-05" } });
        let res = initialize_result(&msg);
        assert_eq!(res["protocolVersion"], "2024-11-05");
        assert_eq!(res["serverInfo"]["name"], "xerj-mcp");
    }

    #[test]
    fn tools_list_has_all_eleven() {
        let specs = tool_specs();
        let names: Vec<&str> = specs
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(names.len(), 11);
        for n in [
            "xerj_search",
            "xerj_semantic_search",
            "xerj_vector_search",
            "xerj_hybrid_search",
            "xerj_memory_store",
            "xerj_memory_recall",
            "xerj_brain_ego",
            "xerj_brain_link",
            "xerj_brain_unlink",
            "xerj_brain_overview",
            // #977: reference-coding retrieval, served by the same pipeline
            // as `xerj code` (LAST in the list, by design).
            "xerj_code_search",
        ] {
            assert!(names.contains(&n), "missing tool {n}");
        }
        // Deliberately NOT exposed: raw traversal, bulk dumps, operator actions.
        for forbidden in ["graph_expand", "autoindex", "dump"] {
            assert!(
                !names.iter().any(|n| n.contains(forbidden)),
                "must not expose {forbidden}"
            );
        }
    }

    #[test]
    fn every_brain_tool_carries_both_honesty_strings() {
        let specs = tool_specs();
        for tool in specs.as_array().unwrap() {
            let name = tool["name"].as_str().unwrap();
            if !name.starts_with("xerj_brain_") {
                continue;
            }
            let desc = tool["description"].as_str().unwrap();
            assert!(
                desc.contains("not neural understanding"),
                "{name} missing the determinism honesty string"
            );
            assert!(
                desc.contains("not a graph database"),
                "{name} missing the not-a-graph-database honesty string"
            );
        }
    }

    // ── loopback admin-key fallback (#961) ──────────────────────────────

    /// Serialises the tests that move process-global state. Discovery reads
    /// the working directory and `HOME`, both per-process.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A temporary working directory and `HOME`, restored on drop (including
    /// on panic) so the discovery tests see only the key files they wrote —
    /// the same sandbox shape as `xerj-autoindex`'s cli tests, where the
    /// helper this exercises was born.
    struct Sandbox {
        _lock: std::sync::MutexGuard<'static, ()>,
        dir: tempfile::TempDir,
        previous_dir: std::path::PathBuf,
        previous_home: Option<std::ffi::OsString>,
    }

    impl Sandbox {
        fn new() -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            let dir = tempfile::tempdir().unwrap();
            let previous_dir = std::env::current_dir().unwrap();
            let previous_home = std::env::var_os("HOME");
            std::env::set_current_dir(dir.path()).unwrap();
            std::env::set_var("HOME", dir.path().join("home"));
            Self {
                _lock: lock,
                dir,
                previous_dir,
                previous_home,
            }
        }

        fn write(&self, relative: &str, contents: &str) {
            let path = self.dir.path().join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }
    }

    impl Drop for Sandbox {
        fn drop(&mut self) {
            std::env::set_current_dir(&self.previous_dir).unwrap();
            match &self.previous_home {
                Some(home) => std::env::set_var("HOME", home),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    /// The gap #961 is about: `xerj init` writes only XERJ_URL, auth is on by
    /// default, and the documented setup 401'd on every tool call. With a
    /// loopback URL and a key on disk, discovery closes that gap.
    #[test]
    fn loopback_url_with_no_auth_picks_up_the_local_admin_key() {
        let sandbox = Sandbox::new();
        sandbox.write("data/admin.key", "local-admin-key\n");
        let (auth, from) = resolve_auth(None, None, "http://localhost:9200");
        assert_eq!(auth.as_deref(), Some("ApiKey local-admin-key"));
        assert_eq!(
            from,
            Some(sandbox.dir.path().join("data/admin.key")),
            "the announced path is the file actually used"
        );

        // The no-configuration shape: XERJ_URL unset means the loopback
        // default, which is exactly the `xerj init` entry.
        let (auth, _) = resolve_auth(None, None, DEFAULT_XERJ_URL);
        assert_eq!(auth.as_deref(), Some("ApiKey local-admin-key"));
    }

    /// Discovery is loopback-only. A locally readable admin key sent to a
    /// host that did not write it is a credential leak — including via the
    /// userinfo trick, where everything before `@` is credentials and the
    /// host is `evil.com`.
    #[test]
    fn non_loopback_urls_never_receive_a_discovered_key() {
        let sandbox = Sandbox::new();
        sandbox.write("data/admin.key", "local-admin-key\n");
        for url in [
            "http://search.example.com:9200",
            "http://192.168.1.5:9200",
            "https://localhost.evil.com:9200",
            "http://localhost:9200@evil.com/",
            "http://127.0.0.1:80@evil.com/",
            "localhost:9200@evil.com",
        ] {
            let (auth, from) = resolve_auth(None, None, url);
            assert!(
                auth.is_none() && from.is_none(),
                "{url} must never be sent a key found on this disk"
            );
        }
    }

    /// A credential the user supplied is never overwritten by one found on
    /// disk — and an *empty* value is unset, not a credential: shadowing
    /// discovery with `Authorization: ` would reproduce the silent 401.
    #[test]
    fn explicit_auth_wins_over_loopback_discovery() {
        let sandbox = Sandbox::new();
        sandbox.write("data/admin.key", "local-admin-key\n");

        let (auth, from) = resolve_auth(
            Some("Bearer token".into()),
            Some("ApiKey env".into()),
            "http://localhost:9200",
        );
        assert_eq!(auth.as_deref(), Some("Bearer token"));
        assert!(
            from.is_none(),
            "an explicit --auth is never re-sourced from disk"
        );

        let (auth, from) = resolve_auth(None, Some("ApiKey env".into()), "http://localhost:9200");
        assert_eq!(auth.as_deref(), Some("ApiKey env"));
        assert!(from.is_none(), "XERJ_AUTH also wins over discovery");

        let (auth, _) = resolve_auth(Some(String::new()), None, "http://localhost:9200");
        assert_eq!(
            auth.as_deref(),
            Some("ApiKey local-admin-key"),
            "an empty --auth/XERJ_AUTH must fall through to discovery"
        );
    }

    /// A server that has not finished writing its key, or a file truncated
    /// by hand, must not become `Authorization: ApiKey ` — the same 401 with
    /// none of the diagnosis.
    #[test]
    fn empty_key_files_are_skipped_not_turned_into_empty_headers() {
        let sandbox = Sandbox::new();
        sandbox.write("data/admin.key", "  \n\t\n");
        sandbox.write("xerj-data/admin.key", "");
        let (auth, from) = resolve_auth(None, None, "http://localhost:9200");
        assert!(
            auth.is_none() && from.is_none(),
            "whitespace-only and zero-byte files are not credentials"
        );

        sandbox.write("home/.xerj/admin.key", "real-key\n");
        let (auth, _) = resolve_auth(None, None, "http://localhost:9200");
        assert_eq!(
            auth.as_deref(),
            Some("ApiKey real-key"),
            "an empty candidate must be skipped over, not stop the search"
        );
    }
}
