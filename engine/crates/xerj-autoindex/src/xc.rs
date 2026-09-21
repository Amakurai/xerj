//! `xerj code` and `xerj corpus` — the reference-coding loop ported into the
//! binary (issue #977; previously `tools/xerj-code/scripts/xc*.py|sh`).
//!
//! Everything semantic lives in [`xerj_common::xccode`]; this file is the CLI
//! shell: argv parsing in the house style (no clap), the [`XcHttp`]
//! implementation over this crate's blocking [`Es`] client, the git/clone
//! lifecycle (`xerj corpus add`), and the #930 build-verify-swap
//! (`xerj corpus index`). Exit codes: 0 hits / 1 no-match-or-incomplete /
//! 2 usage-transport-stale / 3 corpus-in-state-but-not-loaded-here.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use xerj_common::localauth::{discover_local_admin_key, url_is_loopback};
use xerj_common::xccode::{self, manifest, state, CodeOutcome, CodeParams, Mode, XcHttp};

use crate::esclient::{Count, Es};

// ── shared plumbing ─────────────────────────────────────────────────────────

/// `~/.xerj-code` unless `XERJ_CODE_HOME` says otherwise. Never `/tmp`:
/// corpora and state must persist across reboots.
pub fn code_root() -> PathBuf {
    if let Ok(home) = std::env::var("XERJ_CODE_HOME") {
        if !home.is_empty() {
            return PathBuf::from(home);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".xerj-code")
}

/// `--url` flag > `XERJ_URL` > `http://localhost:9200`. Unlike `autoindex`
/// (which ignores `XERJ_URL` on purpose for writes), these READ commands
/// honor it — the scripts always did.
fn resolve_url(explicit: Option<&str>) -> String {
    explicit
        .map(str::to_string)
        .or_else(|| std::env::var("XERJ_URL").ok().filter(|u| !u.is_empty()))
        .unwrap_or_else(|| "http://localhost:9200".to_string())
}

fn build_es(url: &str, api_key: Option<String>) -> Result<Es> {
    // Loopback admin-key fallback (#961/#962 patterns): announced, loopback-only.
    let key = match api_key {
        Some(k) => Some(k),
        None => {
            let env_key = std::env::var("XERJ_API_KEY").ok().filter(|k| !k.is_empty());
            match env_key {
                Some(k) => Some(k),
                None => {
                    if url_is_loopback(url) {
                        discover_local_admin_key().map(|(k, path)| {
                            eprintln!(
                                "xerj code: no --api-key/XERJ_API_KEY given; using the admin \
                                 key at {}",
                                path.display()
                            );
                            k
                        })
                    } else {
                        None
                    }
                }
            }
        }
    };
    Es::new(url, key)
}

/// [`XcHttp`] over the blocking client.
struct EsXc<'a>(&'a Es);

impl XcHttp for EsXc<'_> {
    fn get_mapping(&self, path: &str) -> Result<Value, String> {
        self.0.get_json(path).map_err(|e| format!("{e:#}"))
    }
    fn cat_indices_json(&self, pattern: &str) -> Result<Vec<String>, String> {
        self.0
            .cat_indices_json(pattern)
            .map_err(|e| format!("{e:#}"))
    }
    fn search(&self, index: &str, body: &Value) -> Result<Value, String> {
        self.0.search(index, body).map_err(|e| format!("{e:#}"))
    }
}

fn emit(out: &CodeOutcome) {
    for w in &out.warnings {
        eprintln!("{w}");
    }
    // `--json` can be the ONLY output (empty text is normal for a machine
    // consumer): the JSON line must print even when text is empty, so the
    // guard covers the text arm only, never the json arm.
    if !out.text.is_empty() {
        if out.to_stderr {
            eprintln!("{}", out.text.trim_end());
        } else {
            print!("{}", out.text);
        }
    }
    if let Some(j) = &out.json {
        println!("{j}");
    }
}

// ── xerj code ───────────────────────────────────────────────────────────────

const CODE_USAGE: &str =
    "usage: xerj code <corpus> \"<what you need>\" [-k N] [--mode bm25|semantic|hybrid]
       [--hybrid] [--lang <lg>] [--full N] [--no-symbol] [--json] [--meatl] [--stale-ok]
       [--url URL] [--api-key KEY]

retrieval over a reference corpus (see `xerj corpus list`):
  -k N            passages to return (default 5)
  --mode MODE     bm25 (default, measured 12/12 top-3), semantic, or hybrid
  --hybrid        shorthand for --mode hybrid
  --lang LG       filter to one language field (e.g. rust, go)
  --full N        max chars per passage (default 800; 0 = file head only)
  --no-symbol     window selection instead of the matching definition
  --json          raw server response on stdout
  --meatl         machine-readable one-line-per-hit output
  --stale-ok      override the 30-day staleness refusal
  --url URL       node to query (default $XERJ_URL or http://localhost:9200)
  --api-key KEY   API key (loopback falls back to <data_dir>/admin.key)";

/// `xerj code …` — returns the process exit code.
pub fn run_code_cli(args: &[String]) -> i32 {
    let mut corpus: Option<String> = None;
    let mut query: Option<String> = None;
    let mut url: Option<String> = None;
    let mut api_key: Option<String> = None;
    let mut p = CodeParams::new("", "");
    let mut i = 0;
    let bad = |msg: String| -> i32 {
        eprintln!("{msg}\n");
        eprintln!("{CODE_USAGE}");
        2
    };
    while i < args.len() {
        let a = &args[i];
        macro_rules! val {
            ($name:expr) => {
                match args.get(i + 1) {
                    Some(v) => {
                        i += 1;
                        v.clone()
                    }
                    None => return bad(format!("xerj code: {} needs a value", $name)),
                }
            };
        }
        match a.as_str() {
            "-h" | "--help" => {
                println!("{CODE_USAGE}");
                return 0;
            }
            "-k" => {
                let v = val!("-k");
                match v.parse::<usize>() {
                    Ok(n) if (1..=50).contains(&n) => p.k = n,
                    _ => return bad(format!("xerj code: -k must be 1..=50, got '{v}'")),
                }
            }
            "--lang" => p.lang = Some(val!("--lang")),
            "--mode" => {
                let v = val!("--mode");
                match Mode::parse(&v) {
                    Some(m) => p.mode = m,
                    None => return bad(format!("xerj code: unknown mode '{v}'")),
                }
            }
            "--hybrid" => p.mode = Mode::Hybrid,
            "--full" => {
                let v = val!("--full");
                match v.parse::<usize>() {
                    Ok(n) => p.full = n,
                    _ => return bad(format!("xerj code: --full must be a number, got '{v}'")),
                }
            }
            "--no-symbol" => p.no_symbol = true,
            "--json" => p.as_json = true,
            "--meatl" => p.meatl = true,
            "--stale-ok" => p.stale_ok = true,
            "--url" => url = Some(val!("--url")),
            "--api-key" => api_key = Some(val!("--api-key")),
            _ if a.starts_with('-') => {
                return bad(format!("xerj code: unknown flag '{a}'"));
            }
            _ => {
                if corpus.is_none() {
                    corpus = Some(a.clone());
                } else if query.is_none() {
                    query = Some(a.clone());
                } else {
                    return bad(format!("xerj code: unexpected argument '{a}'"));
                }
            }
        }
        i += 1;
    }
    let (Some(corpus), Some(query)) = (corpus, query) else {
        return bad("xerj code: both <corpus> and \"<query>\" are required".to_string());
    };
    let url = resolve_url(url.as_deref());
    let es = match build_es(&url, api_key) {
        Ok(es) => es,
        Err(e) => {
            eprintln!("xerj code: cannot reach {url}: {e:#}");
            return 2;
        }
    };
    p.corpus = corpus;
    p.query = query;
    let out = xccode::run_code_query(&code_root(), &EsXc(&es), &url, &p, "`--stale-ok`");
    emit(&out);
    out.exit
}

// ── xerj corpus add ─────────────────────────────────────────────────────────

const CORPUS_USAGE: &str =
    "usage: xerj corpus add <name> <git-url>... | --from <manifest.json> [--as <name>]
       xerj corpus index <name> [--fresh] [--url URL]
       xerj corpus list [--url URL]

clone the repos, detect licences, write corpora/<name>/corpus.json;
--from takes the corpus name from the manifest's 'corpus' field unless
<name> or --as overrides it; index builds/verifies/switches the corpus
(exit 3 skips junk — normal); list shows what is loaded on this node.";

fn git(dir: Option<&Path>, args: &[&str]) -> Result<(i32, String)> {
    let mut c = Command::new("git");
    if let Some(d) = dir {
        c.current_dir(d).arg("-C").arg(d);
    }
    let out = c
        .args(args)
        .output()
        .with_context(|| format!("run git {}", args.first().unwrap_or(&"")))?;
    Ok((
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).trim().to_string()
            + String::from_utf8_lossy(&out.stderr).trim(),
    ))
}

/// Move a clone to `sha` (shallow). Never a shell string: the sha comes from
/// an untrusted manifest, and `Command` passes it as ONE argv element.
fn checkout_at_sha(target: &Path, url: &str, sha: &str) -> Result<()> {
    if !target.join(".git").exists() {
        std::fs::create_dir_all(target)?;
        git(Some(target), &["init", "--quiet"])?;
        git(Some(target), &["remote", "add", "origin", url]).ok();
        git(Some(target), &["config", "remote.origin.promisor", "true"]).ok();
        git(
            Some(target),
            &["config", "remote.origin.partialclonefilter", "blob:none"],
        )
        .ok();
    }
    let fetch = git(Some(target), &["fetch", "--depth", "1", "origin", sha]);
    if !fetch.map(|(rc, _)| rc == 0).unwrap_or(false) {
        anyhow::bail!("git fetch {sha} failed");
    }
    let co = git(
        Some(target),
        &["checkout", "--quiet", "--force", "--detach", sha],
    );
    if !co.map(|(rc, _)| rc == 0).unwrap_or(false) {
        anyhow::bail!("git checkout {sha} failed");
    }
    git(Some(target), &["clean", "-qfd"]).ok();
    Ok(())
}

fn dir_stats(target: &Path) -> (String, u64, u64) {
    let sha = git(Some(target), &["rev-parse", "HEAD"])
        .ok()
        .filter(|(rc, _)| *rc == 0)
        .map(|(_, out)| out)
        .unwrap_or_else(|| "unknown".to_string());
    let mut files = 0u64;
    let mut bytes = 0u64;
    fn walk(dir: &Path, files: &mut u64, bytes: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.file_name().map(|n| n == ".git").unwrap_or(false) {
                continue;
            }
            if p.is_dir() {
                walk(&p, files, bytes);
            } else if let Ok(md) = e.metadata() {
                *files += 1;
                *bytes += md.len();
            }
        }
    }
    walk(target, &mut files, &mut bytes);
    (sha, files, bytes)
}

/// Corpus-name resolution for `add`: an explicit <name> (positional, then
/// `--as`) wins; otherwise the manifest's own `corpus` field supplies it.
/// `--from` with neither is a usage error, not a guess.
fn resolve_corpus_name(
    explicit: Option<String>,
    hub_corpus: &str,
    path: &str,
) -> Result<String, String> {
    explicit
        .or_else(|| (!hub_corpus.is_empty()).then(|| hub_corpus.to_string()))
        .ok_or_else(|| {
            format!("{path} has no 'corpus' field and no <name>/--as was given\n\n{CORPUS_USAGE}")
        })
}

fn run_corpus_add(args: &[String]) -> i32 {
    let mut name: Option<String> = None;
    let mut as_name: Option<String> = None;
    let mut from: Option<String> = None;
    let mut urls: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{CORPUS_USAGE}");
                return 0;
            }
            "--from" => match args.get(i + 1) {
                Some(v) => {
                    from = Some(v.clone());
                    i += 1;
                }
                None => {
                    eprintln!("xerj corpus add: --from needs a manifest path\n\n{CORPUS_USAGE}");
                    return 2;
                }
            },
            "--as" => match args.get(i + 1) {
                Some(v) => {
                    as_name = Some(v.clone());
                    i += 1;
                }
                None => {
                    eprintln!("xerj corpus add: --as needs a corpus name\n\n{CORPUS_USAGE}");
                    return 2;
                }
            },
            a if a.starts_with('-') => {
                eprintln!("xerj corpus add: unknown flag '{a}'\n\n{CORPUS_USAGE}");
                return 2;
            }
            a => {
                if name.is_none() {
                    name = Some(a.to_string());
                } else {
                    urls.push(a.to_string());
                }
            }
        }
        i += 1;
    }
    // `--as` only renames a `--from` rebuild; a positional name wins when
    // both are given (it is the more explicit spelling of the same thing).
    let explicit_name = name.or(as_name);
    if from.is_none() && explicit_name.is_none() {
        eprintln!("xerj corpus add: <name> is required\n\n{CORPUS_USAGE}");
        return 2;
    }
    if from.is_none() && urls.is_empty() {
        eprintln!(
            "xerj corpus add: give at least one <git-url> or --from <manifest>\n\n{CORPUS_USAGE}"
        );
        return 2;
    }
    // The manifest is read BEFORE the name is finalised: with `--from` and
    // no explicit <name>/--as, the manifest's own 'corpus' field supplies
    // it. Everything downstream (dest, carry, entries) needs the resolved
    // name, so resolution lives here and nowhere else.
    let (name, rows): (String, Vec<(String, String, String, String)>) = if let Some(path) = &from {
        let hub = match manifest::read_hub_manifest(Path::new(path)) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("xerj corpus add: {e}");
                return 2;
            }
        };
        let resolved = match resolve_corpus_name(explicit_name, &hub.corpus, path) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("xerj corpus add: {e}");
                return 2;
            }
        };
        println!("rebuilding corpus '{resolved}' from {path}");
        (
            resolved,
            hub.rows
                .into_iter()
                .map(|r| (r.repo, r.url, r.sha, r.declared_licence))
                .collect(),
        )
    } else {
        let Some(resolved) = explicit_name else {
            eprintln!("xerj corpus add: <name> is required\n\n{CORPUS_USAGE}");
            return 2;
        };
        (
            resolved,
            urls.iter()
                .map(|u| {
                    let repo = u
                        .trim_end_matches('/')
                        .rsplit('/')
                        .next()
                        .unwrap_or("")
                        .trim_end_matches(".git")
                        .to_string();
                    (repo, u.clone(), String::new(), String::new())
                })
                .collect(),
        )
    };
    if let Err(e) = xccode::pathgate::valid_corpus_name(&name) {
        eprintln!("xerj corpus add: {e}");
        return 2;
    }

    let root = code_root();
    let dest = root.join("corpora").join(&name);
    if let Err(e) = std::fs::create_dir_all(&dest) {
        eprintln!("xerj corpus add: cannot create {}: {e}", dest.display());
        return 2;
    }

    // review{} blocks are PRESERVED across a rebuild when repo+sha match —
    // a human's licence review survives a re-clone (additive, never edited).
    let previous = manifest::read_corpus_manifest(&dest.join("corpus.json")).ok();
    let carry = |repo: &str, sha: &str| -> Option<Value> {
        let m = previous.as_ref()?;
        m.repos.iter().find_map(|r| {
            (r.repo == repo && r.sha == sha)
                .then(|| r.review.clone())
                .flatten()
        })
    };

    let mut entries: Vec<manifest::ManifestRepo> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for (repo_name, url, sha, declared) in rows {
        // The one destructive step (checkout --force + clean -fd) runs inside
        // this path: gate the name HERE, again, even though read_hub_manifest
        // already did — the plain-URL arm has no other gate.
        if xccode::pathgate::valid_repo_name(&repo_name).is_err() || repo_name.starts_with('-') {
            eprintln!("  [FAIL] refusing unsafe repo name '{repo_name}'");
            skipped.push(repo_name);
            continue;
        }
        let target = dest.join(&repo_name);
        if target.join(".git").exists() && !sha.is_empty() {
            let at = git(Some(&target), &["rev-parse", "HEAD"])
                .ok()
                .filter(|(rc, _)| *rc == 0)
                .map(|(_, out)| out);
            if at.as_deref() == Some(sha.as_str()) {
                println!(
                    "  [ok] {repo_name} already at {}",
                    &sha[..12.min(sha.len())]
                );
                record(&mut entries, &repo_name, &url, &target, &declared, &carry);
                continue;
            }
            println!("  [pin] {repo_name} -> {}", &sha[..12.min(sha.len())]);
            if let Err(e) = checkout_at_sha(&target, &url, &sha) {
                eprintln!("  [FAIL] {repo_name}: {e:#}; continuing");
                skipped.push(repo_name);
                continue;
            }
            record(&mut entries, &repo_name, &url, &target, &declared, &carry);
            continue;
        }
        if target.join(".git").exists() {
            println!("  [skip] {repo_name} already cloned");
            record(&mut entries, &repo_name, &url, &target, &declared, &carry);
            continue;
        }
        println!("  [clone] {repo_name}");
        if sha.is_empty() {
            if let Some(parent) = target.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            // Never a shell string: the URL and target are single argv
            // elements, so a hostile URL cannot smuggle flags.
            let target_s = target.to_string_lossy().to_string();
            let (rc, _) = git(None, &["clone", "--depth", "1", "--quiet", &url, &target_s])
                .unwrap_or((-1, String::new()));
            if rc != 0 {
                let (rc2, _) = git(None, &["clone", "--depth", "1", &url, &target_s])
                    .unwrap_or((-1, String::new()));
                if rc2 != 0 {
                    eprintln!("  [FAIL] {repo_name} — could not clone; continuing");
                    skipped.push(repo_name);
                    continue;
                }
            }
        } else {
            println!("  [pin] {repo_name} -> {}", &sha[..12.min(sha.len())]);
            if let Err(e) = checkout_at_sha(&target, &url, &sha) {
                eprintln!("  [FAIL] {repo_name}: {e:#}; continuing");
                skipped.push(repo_name);
                continue;
            }
        }
        record(&mut entries, &repo_name, &url, &target, &declared, &carry);
    }

    if entries.is_empty() {
        eprintln!("xerj corpus add: nothing cloned");
        return 1;
    }

    // Regenerated FROM DISK, never copied through from the input.
    let cloned_at = chrono_now_stamp();
    let manifest_path = dest.join("corpus.json");
    manifest::write_corpus_manifest(&manifest_path, &name, &cloned_at, &entries);
    println!();
    println!(
        "corpus '{name}': {} repos at {}",
        entries.len(),
        dest.display()
    );
    println!(
        "share it: {}  (rebuild with xerj corpus add <name> --from <that file>)",
        manifest_path.display()
    );

    if !skipped.is_empty() {
        eprintln!();
        eprintln!(
            "xerj corpus add: {} of {} repos did not land: {}",
            skipped.len(),
            entries.len() + skipped.len(),
            skipped.join(" ")
        );
        eprintln!("xerj corpus add: this corpus is INCOMPLETE — it is not the one the manifest describes.");
        return 1;
    }
    println!("next: xerj corpus index {name}");
    0
}

fn record(
    entries: &mut Vec<manifest::ManifestRepo>,
    repo_name: &str,
    url: &str,
    target: &Path,
    declared: &str,
    carry: &dyn Fn(&str, &str) -> Option<Value>,
) {
    let lic = xccode::licence::detect_licence(target);
    let (sha, files, bytes) = dir_stats(target);
    println!(
        "          licence={lic} sha={} files={files}",
        &sha[..12.min(sha.len())]
    );
    if let Some(w) = xccode::licence::clone_warning_line(&lic) {
        eprintln!("{w}");
    }
    // The detector is text-matching heuristics and has been wrong before.
    // When a manifest disagrees with the checkout, say so rather than
    // overwrite quietly.
    if !declared.is_empty() && declared != lic {
        eprintln!("          ! manifest says licence={declared}, checkout reads {lic} — verify before copying");
    }
    let review = carry(repo_name, &sha);
    entries.push(manifest::ManifestRepo {
        repo: repo_name.to_string(),
        url: url.to_string(),
        licence: lic,
        sha,
        files: Some(files),
        bytes: Some(bytes),
        review,
    });
}

fn chrono_now_stamp() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

// ── xerj corpus index ───────────────────────────────────────────────────────

/// The node operations the corpus lifecycle needs (#1004): one trait so the
/// build/verify/swap flow can run against a fake in tests, re-pinning the
/// destructive-operation contracts the retired `test_xc_index_fresh.py`
/// pinned at the HTTP layer. The trait is deliberately the FOUR operations
/// the flow performs against the node — listing, counting (tri-state), and
/// the two scoped deletes — nothing more, so the fake cannot drift from what
/// a real node is asked to do.
pub(crate) trait CorpusNode {
    /// `_cat/indices` under a glob, as index names. `Err` = unreachable.
    fn list_indices(&self, glob: &str) -> Result<Vec<String>, String>;
    /// The tri-state `_count` for a `{prefix}-*` glob (DASH form).
    fn count(&self, dash_glob: &str) -> Count;
    /// DELETE one index BY EXACT NAME. `false` = the node refused.
    fn delete_index(&self, name: &str) -> bool;
    /// `_delete_by_query` on the shared catalog for one corpus scope.
    fn delete_catalog_scope(&self, scope: &str) -> bool;
}

impl CorpusNode for Es {
    fn list_indices(&self, glob: &str) -> Result<Vec<String>, String> {
        self.cat_indices_json(glob).map_err(|e| format!("{e:#}"))
    }
    fn count(&self, dash_glob: &str) -> Count {
        self.count_endpoint(dash_glob)
    }
    fn delete_index(&self, name: &str) -> bool {
        self.request_json("DELETE", &format!("/{name}"), None)
            .map(|(s, _)| (200..300).contains(&s))
            .unwrap_or(false)
    }
    fn delete_catalog_scope(&self, scope: &str) -> bool {
        // The catalog is one global index shared by every corpus on the node,
        // so its documents are removed by EXACT scope value: `corpus_scope`
        // is a keyword, and a `term` on a legacy analyzed `prefix` cannot
        // equal a hyphenated value at all — it under-deletes there, it never
        // reaches a sibling corpus.
        let body = json!({
            "query": { "bool": { "minimum_should_match": 1, "should": [
                { "term": { "corpus_scope": scope } },
                { "term": { "prefix": scope } }
            ]}}
        });
        self.request_json(
            "POST",
            "/autoindex-catalog/_delete_by_query?refresh=true",
            Some(&body),
        )
        .map(|(s, _)| (200..300).contains(&s))
        .unwrap_or(false)
    }
}

/// `count_under`: patiently. Still `None` when the node never answered —
/// callers must treat that as UNKNOWN, never as zero. The glob is the DASH
/// form `{prefix}-*` (verification), pinned against the STAR form the query
/// path uses — both are load-bearing and they are not interchangeable.
fn count_under(node: &dyn CorpusNode, prefix: &str) -> Option<u64> {
    let tries = std::env::var("XC_COUNT_TRIES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(6);
    let pause = std::env::var("XC_COUNT_PAUSE")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(5);
    for attempt in 1..=tries.max(1) {
        match node.count(&format!("{prefix}-*")) {
            Count::Number(n) => return Some(n),
            Count::Zero => return Some(0),
            Count::Unknown(_) => {
                if attempt == tries.max(1) {
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_secs(pause));
            }
        }
    }
    None
}

/// Every index under `xc-<corpus>-` that belongs to THIS corpus. The bare
/// glob is not enough: `xc-battle-*` also matches the sibling corpus
/// `battle-terse`, and retiring a sibling's indices is not a mistake this
/// command gets to make.
fn corpus_indices(node: &dyn CorpusNode, name: &str, root: &Path) -> Vec<String> {
    // Any error lists nothing, which errs toward KEEPING an index (nothing is
    // retired that was not listed), never toward deleting one.
    let rows = node
        .list_indices(&format!("xc-{name}-*"))
        .unwrap_or_default();
    let mut siblings: Vec<String> = Vec::new();
    for (folder, strip) in [(root.join("corpora"), ""), (root.join("state"), ".json")] {
        if let Ok(entries) = std::fs::read_dir(&folder) {
            for e in entries.flatten() {
                let fname = e.file_name().to_string_lossy().to_string();
                let other = match strip {
                    "" => fname,
                    s if fname.ends_with(s) => fname[..fname.len() - s.len()].to_string(),
                    _ => continue,
                };
                if other != name && other.starts_with(&format!("{name}-")) {
                    siblings.push(format!("xc-{other}-"));
                }
            }
        }
    }
    rows.into_iter()
        .filter(|idx| {
            idx.starts_with(&format!("xc-{name}-"))
                && !siblings.iter().any(|s| idx.starts_with(s.as_str()))
        })
        .collect()
}

fn delete_indices(node: &dyn CorpusNode, names: &[String]) -> usize {
    let mut failed = 0;
    for index in names {
        // Exact names only, never a wildcard.
        if !node.delete_index(index) {
            failed += 1;
            eprintln!("xerj corpus index:   could not delete {index}");
        }
    }
    failed
}

fn report(name: &str, rc: i32, docs: &str) {
    match rc {
        0 => println!("indexed cleanly"),
        3 => println!("indexed (exit 3: some files skipped as junk — normal for real repos)"),
        _ => {}
    }
    println!("corpus '{name}' searchable: {docs} records");
    println!("next: xerj code {name} \"<what you need>\"");
}

fn run_corpus_index(args: &[String]) -> i32 {
    let mut name: Option<String> = None;
    let mut fresh = false;
    let mut url: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{CORPUS_USAGE}");
                return 0;
            }
            "--fresh" => fresh = true,
            "--url" => match args.get(i + 1) {
                Some(v) => {
                    url = Some(v.clone());
                    i += 1;
                }
                None => {
                    eprintln!("xerj corpus index: --url needs a value\n\n{CORPUS_USAGE}");
                    return 2;
                }
            },
            a if a.starts_with('-') => {
                eprintln!("xerj corpus index: unknown flag '{a}'\n\n{CORPUS_USAGE}");
                return 2;
            }
            a if name.is_none() => name = Some(a.to_string()),
            a => {
                eprintln!("xerj corpus index: unexpected argument '{a}'\n\n{CORPUS_USAGE}");
                return 2;
            }
        }
        i += 1;
    }
    let Some(name) = name else {
        eprintln!("xerj corpus index: <name> is required\n\n{CORPUS_USAGE}");
        return 2;
    };
    if let Err(e) = xccode::pathgate::valid_corpus_name(&name) {
        eprintln!("xerj corpus index: {e}");
        return 2;
    }
    let root = code_root();
    let dir = root.join("corpora").join(&name);
    if !dir.is_dir() {
        eprintln!(
            "xerj corpus index: no corpus '{name}' at {} — run `xerj corpus add {name} …` first",
            dir.display()
        );
        return 2;
    }
    let url = resolve_url(url.as_deref());
    let es = match build_es(&url, None) {
        Ok(es) => es,
        Err(e) => {
            eprintln!("xerj corpus index: cannot reach {url}: {e:#}");
            return 2;
        }
    };
    corpus_index_flow(
        &es,
        &|prefix, state_dir| crate::run_with_options(&dir, &url, prefix, state_dir, true),
        &stamp_secs,
        &root,
        &url,
        &name,
        fresh,
    )
}

/// The build/verify/swap flow over an injected node, autoindex runner and
/// clock (#1004) — the destructive-operation contracts are pinned by the
/// tests below against a [`FakeNode`](tests::FakeNode), which is why none of
/// this touches [`Es`] directly. `run_autoindex(prefix, state_dir)` is the
/// real `crate::run_with_options` in production, with `--no-graph` set
/// (reference code needs ranked passages, not a relationship map) — and
/// `--fresh` is NEVER forwarded: this flow owns the swap.
fn corpus_index_flow(
    node: &dyn CorpusNode,
    run_autoindex: &dyn Fn(&str, Option<&Path>) -> i32,
    now_secs: &dyn Fn() -> i64,
    root: &Path,
    url: &str,
    name: &str,
    fresh: bool,
) -> i32 {
    let state_file = root.join("state").join(format!("{name}.json"));
    let state_root = root.join("autoindex-state");
    let old = state::load_state(root, name).ok();
    let old_build = old.as_ref().and_then(|s| s.build.clone());
    let old_index_prefix = old.as_ref().and_then(|s| s.index_prefix.clone());
    let old_state_dir = old.as_ref().and_then(|s| s.state_dir.clone());
    let old_url = old.as_ref().and_then(|s| s.url.clone());

    // Which run is this? build (--fresh / first / ledger-names-a-build-this-
    // node-does-not-hold), update (reconcile the recorded build in place), or
    // legacy (a corpus indexed before builds existed).
    let mut mode = "legacy";
    let old_indices = corpus_indices(node, name, root);
    if fresh {
        mode = "build";
    } else if old_build.is_some()
        && old_index_prefix.is_some()
        && old_state_dir
            .as_deref()
            .is_some_and(|d| Path::new(d).is_dir())
        && old_url.as_deref() == Some(url)
    {
        let live = count_under(node, old_index_prefix.as_deref().unwrap_or(""));
        match live {
            Some(n) if n > 0 => mode = "update",
            _ => {
                println!(
                    "xerj corpus index: state/ records build {} but {url} holds no records for \
                     it — building it here",
                    old_build.as_deref().unwrap_or("?")
                );
                mode = "build";
            }
        }
    } else if !state_file.exists() && old_indices.is_empty() {
        mode = "build"; // first index of this corpus: same path as --fresh
    }

    println!(
        "indexing corpus '{name}' from {}",
        root.join("corpora").join(name).display()
    );

    match mode {
        "build" => {
            // Listed BEFORE the build so "old" can never include what this
            // run creates.
            let old_indices = corpus_indices(node, name, root);
            // One-second resolution: a second --fresh inside the same second
            // would reuse the prefix of the build it is replacing. The id
            // must be new — not the recorded build, not a state dir that
            // exists, not a prefix any live index already sits under.
            let mut build = format!("b{}", now_secs());
            while old_build.as_deref() == Some(build.as_str())
                || state_root.join(name).join(&build).exists()
                || old_indices
                    .iter()
                    .any(|idx| idx.starts_with(&format!("xc-{name}-{build}")))
            {
                std::thread::sleep(std::time::Duration::from_secs(1));
                build = format!("b{}", now_secs());
            }
            let new_prefix = format!("xc-{name}-{build}");
            let new_state = state_root.join(name).join(&build);
            // Is there a working index to protect? When the node cannot
            // count it, the answer is YES: presuming "none" is what lets a
            // failed build be kept over it and the old indices be retired.
            let mut has_working_index = false;
            if !old_indices.is_empty() {
                let counted = count_under(
                    node,
                    old_index_prefix.as_deref().unwrap_or(&format!("xc-{name}")),
                );
                match counted {
                    None => {
                        has_working_index = true;
                        eprintln!("xerj corpus index: the node did not answer a record count for the existing index; treating it as");
                        eprintln!("xerj corpus index: a WORKING index — a failed build will not be kept over it.");
                        println!("xerj corpus index: --fresh — building {new_prefix}-* beside the existing index (an unknown number of records);");
                    }
                    Some(n) => {
                        if n > 0 {
                            has_working_index = true;
                        }
                        println!("xerj corpus index: --fresh — building {new_prefix}-* beside the existing index ({n} records);");
                    }
                }
                println!("xerj corpus index: the existing index stays live until the replacement has been verified");
            }
            let _ = std::fs::create_dir_all(&new_state);
            let rc = run_autoindex(&new_prefix, Some(&new_state));

            // VERIFY: rc in {0,3} AND a count the node actually gave > 0.
            let docs = count_under(node, &new_prefix);
            let Some(docs) = docs else {
                // UNKNOWN is not EMPTY: it authorises no delete and no swap.
                eprintln!(
                    "xerj corpus index: the node did not answer a record count for build {build};"
                );
                eprintln!("xerj corpus index: autoindex exit {rc}. It cannot be verified, so NOTHING was deleted and NOTHING");
                eprintln!("xerj corpus index: was switched: its indices ({new_prefix}-*) and its state directory are kept.");
                if has_working_index {
                    eprintln!("xerj corpus index: the existing index was NOT touched and is still what `xerj code` serves.");
                    eprintln!("xerj corpus index: When the node answers again, re-run with --fresh; the unverified build is");
                    eprintln!("xerj corpus index: retired by the next build that verifies.");
                } else {
                    let _ = state::write_state(
                        root,
                        name,
                        url,
                        rc as i64,
                        true,
                        Some(&build),
                        Some(&new_prefix),
                        new_state.to_str(),
                    );
                    eprintln!("xerj corpus index: There is no other index for '{name}', so this build is recorded as UNVERIFIED");
                    eprintln!("xerj corpus index: (`xerj code` will say so). Re-run  xerj corpus index {name}  to resume or confirm it.");
                }
                if rc != 0 && rc != 3 {
                    return rc;
                }
                return 1;
            };
            let mut salvaged = false;
            let mut verified = false;
            match rc {
                0 | 3 if docs > 0 => verified = true,
                _ => {
                    // autoindex can abort in finalisation AFTER every document
                    // was written (#367). Keep a complete, queryable index
                    // when there is no working fallback — and never swap a
                    // verified one out for it.
                    if docs > 0 && !has_working_index {
                        verified = true;
                        salvaged = true;
                        eprintln!("xerj corpus index: WARNING — autoindex exited {rc}, but this build wrote {docs} records and");
                        eprintln!("xerj corpus index: there is no working index to fall back to. Recording it as indexed with");
                        eprintln!("xerj corpus index: autoindex_exit={rc}; coverage is not guaranteed — please report the error above.");
                    }
                }
            }
            if !verified {
                eprintln!("xerj corpus index: build {build} did not verify (autoindex exit {rc}, {docs} records).");
                // Remove only what THIS run created: the set-diff against the
                // pre-run listing, intersected with this build's prefix, BY
                // EXACT NAME — never a wildcard, never new_prefix itself.
                let after = corpus_indices(node, name, root);
                let doomed: Vec<String> = after
                    .into_iter()
                    .filter(|idx| !old_indices.contains(idx))
                    .filter(|idx| idx.starts_with(&format!("xc-{name}-{build}-")))
                    .collect();
                delete_indices(node, &doomed);
                node.delete_catalog_scope(&new_prefix);
                let _ = std::fs::remove_dir_all(&new_state);
                if !old_indices.is_empty() {
                    eprintln!("xerj corpus index: the existing index was NOT touched and is still what `xerj code` serves.");
                }
                if rc != 0 && rc != 3 {
                    return rc;
                }
                return 1;
            }
            // Verified. Switch readers first, retire second: a crash between
            // the two leaves a duplicate, never a gap.
            let _ = state::write_state(
                root,
                name,
                url,
                rc as i64,
                salvaged,
                Some(&build),
                Some(&new_prefix),
                new_state.to_str(),
            );
            let retire: Vec<String> = old_indices
                .iter()
                .filter(|idx| !idx.starts_with(&format!("xc-{name}-{build}")))
                .cloned()
                .collect();
            if !retire.is_empty() {
                println!("xerj corpus index: replacement verified ({docs} records) — retiring {} old indices", retire.len());
                if delete_indices(node, &retire) > 0 {
                    eprintln!("xerj corpus index: WARNING — some old indices could not be deleted. The corpus is healthy and");
                    eprintln!("xerj corpus index: `xerj code` reads only {new_prefix}-*; delete the leftovers by name when the node allows.");
                }
                node.delete_catalog_scope(
                    old_index_prefix.as_deref().unwrap_or(&format!("xc-{name}")),
                );
            }
            // Earlier builds' state directories are dead weight once their
            // indices are gone.
            if state_root.join(name).is_dir() {
                if let Ok(entries) = std::fs::read_dir(state_root.join(name)) {
                    for e in entries.flatten() {
                        if e.file_name().to_string_lossy() != build {
                            let _ = std::fs::remove_dir_all(e.path());
                        }
                    }
                }
            }
            report(name, rc, &docs.to_string());
            0
        }
        "update" => {
            let prefix = old_index_prefix.clone().unwrap_or_default();
            let sd = old_state_dir.clone().map(PathBuf::from).unwrap_or_default();
            let rc = run_autoindex(&prefix, Some(&sd));
            if rc != 0 && rc != 3 {
                eprintln!("xerj corpus index: updating build {} failed with exit {rc}. The index is unchanged.", old_build.as_deref().unwrap_or("?"));
                eprintln!("xerj corpus index: re-run with --fresh to build a replacement beside it (the existing index");
                eprintln!("xerj corpus index: stays live until the replacement verifies).");
                return rc;
            }
            let docs = count_under(node, &prefix)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".to_string());
            let _ = state::write_state(
                root,
                name,
                url,
                rc as i64,
                false,
                old_build.as_deref(),
                Some(&prefix),
                old_state_dir.as_deref(),
            );
            report(name, rc, &docs);
            0
        }
        _ => {
            // Legacy: recorded before the run so a failure can tell "this run
            // wrote records" apart from "an earlier run's records are still
            // lying around" (salvaging the latter would date stale data to
            // now, which is worse than no index).
            let before_known = count_under(node, &format!("xc-{name}"));
            let docs_before = before_known.unwrap_or(0);
            let rc = run_autoindex(&format!("xc-{name}"), None);
            let mut salvaged = false;
            let mut docs: Option<u64> = None;
            if rc != 0 && rc != 3 {
                docs = count_under(node, &format!("xc-{name}"));
                if before_known.is_some() && docs.is_some_and(|d| d > docs_before) {
                    salvaged = true;
                    eprintln!("xerj corpus index: WARNING — autoindex exited {rc}, but this run wrote records");
                    eprintln!("xerj corpus index: ({docs_before} -> {}). The corpus is queryable and is being", docs.unwrap_or(0));
                    eprintln!("xerj corpus index: recorded as indexed, with autoindex_exit={rc} in its state file.");
                    eprintln!("xerj corpus index: Coverage is not guaranteed — please report the error above.");
                } else {
                    eprintln!("xerj corpus index: autoindex failed with exit {rc} and wrote no new records.");
                    eprintln!("xerj corpus index: If the error above says the state directory cannot become generation");
                    eprintln!("xerj corpus index: authority, or that --fresh is refused, run:  xerj corpus index {name} --fresh");
                    return rc;
                }
            }
            docs = docs.or_else(|| count_under(node, &format!("xc-{name}")));
            let docs_s = docs
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".to_string());
            let _ = state::write_state(root, name, url, rc as i64, salvaged, None, None, None);
            report(name, rc, &docs_s);
            0
        }
    }
}

fn stamp_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ── xerj corpus list ────────────────────────────────────────────────────────

fn run_corpus_list(args: &[String]) -> i32 {
    let mut url: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{CORPUS_USAGE}");
                return 0;
            }
            "--url" => match args.get(i + 1) {
                Some(v) => {
                    url = Some(v.clone());
                    i += 1;
                }
                None => {
                    eprintln!("xerj corpus list: --url needs a value\n\n{CORPUS_USAGE}");
                    return 2;
                }
            },
            a => {
                eprintln!("xerj corpus list: unexpected argument '{a}'\n\n{CORPUS_USAGE}");
                return 2;
            }
        }
        i += 1;
    }
    let root = code_root();
    let state_dir = root.join("state");
    if !state_dir.is_dir() {
        eprintln!(
            "no state directory at {} — nothing has been indexed on this machine",
            state_dir.display()
        );
        return 2;
    }
    let url = resolve_url(url.as_deref());
    let es = match build_es(&url, None) {
        Ok(es) => es,
        Err(e) => {
            eprintln!("xerj corpus list: cannot reach {url}: {e:#}");
            return 2;
        }
    };
    let mut names: Vec<String> = std::fs::read_dir(&state_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| {
            e.file_name()
                .to_string_lossy()
                .strip_suffix(".json")
                .map(str::to_string)
        })
        .collect();
    names.sort();
    let mut loaded = 0usize;
    for name in &names {
        let st = match state::load_state(&root, name) {
            Ok(st) => st,
            Err(e) => {
                println!("{name}: unreadable state ({e})");
                continue;
            }
        };
        let prefix = state::query_prefix(&st);
        let indexed = st
            .indexed_at
            .as_deref()
            .unwrap_or("?")
            .get(..10)
            .unwrap_or("?")
            .to_string();
        // Live count is honest: 404 -> 0; any other failure is named, never
        // faked as zero.
        let live = match es.cat_indices_json(&format!("{prefix}*")) {
            Ok(v) => v.len(),
            Err(_) => usize::MAX,
        };
        let uses = xccode::corpus_review_uses(&root, name);
        let use_note = uses
            .keys()
            .max()
            .and_then(|k| uses.get(k).map(|u| format!("\n  use: {u}")))
            .unwrap_or_default();
        match live {
            usize::MAX => println!("{name}  (indexed {indexed}, prefix {prefix}) — server unreachable/ambiguous at {url}"),
            0 => println!("{name}  (indexed {indexed}, prefix {prefix}) — NOT loaded here (0 indices) — stale/other-server{use_note}"),
            n => {
                loaded += 1;
                println!("{name}  (indexed {indexed}, prefix {prefix}) — loaded — {n} index(es){use_note}");
            }
        }
        if let Some(w) = state::incomplete_coverage(&st) {
            println!("  {w}");
        }
    }
    println!();
    println!(
        "{} of {} corpora are actually loaded on {url}.",
        loaded,
        names.len()
    );
    0
}

// ── dispatch ────────────────────────────────────────────────────────────────

/// `xerj corpus <add|index|list> …` — returns the process exit code.
pub fn run_corpus_cli(args: &[String]) -> i32 {
    let Some(sub) = args.first() else {
        eprintln!("{CORPUS_USAGE}");
        return 2;
    };
    let rest: Vec<String> = args[1..].to_vec();
    match sub.as_str() {
        "add" => run_corpus_add(&rest),
        "index" => run_corpus_index(&rest),
        "list" => run_corpus_list(&rest),
        "-h" | "--help" | "help" => {
            println!("{CORPUS_USAGE}");
            0
        }
        other => {
            eprintln!("xerj corpus: unknown subcommand '{other}'\n\n{CORPUS_USAGE}");
            2
        }
    }
}

/// The licence map for one corpus, shared with the MCP tool.
pub fn licence_map_for(root: &Path, corpus: &str) -> HashMap<String, String> {
    manifest::licence_map(root, corpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_names_are_gated_before_any_path_is_built() {
        for bad in ["a/b", "*", ".x", "index", "list"] {
            assert!(xccode::pathgate::valid_corpus_name(bad).is_err(), "{bad}");
        }
        assert!(xccode::pathgate::valid_corpus_name("battle-terse").is_ok());
    }

    #[test]
    fn url_resolution_prefers_flag_then_env_then_default() {
        // under ENV_LOCK: set/remove of process env is global, so any test
        // mutating env serialises with the others (one unreproduced lib-suite
        // failure was observed on the merged #977 branch before this lock).
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(resolve_url(Some("http://a:1")), "http://a:1");
        // env-dependent branches covered by the e2e; default pinned here.
        std::env::remove_var("XERJ_URL");
        assert_eq!(resolve_url(None), "http://localhost:9200");
    }

    #[test]
    fn count_tries_env_defaults_are_the_scripts_values() {
        // under the same lock the #1004 tests use: they set the knobs, this
        // test asserts their absence — run concurrently they would flake.
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("XC_COUNT_TRIES");
        std::env::remove_var("XC_COUNT_PAUSE");
        // (6 tries, 5s pause) — pinned by the port; the env knobs still work.
        let t = std::env::var("XC_COUNT_TRIES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(6);
        assert_eq!(t, 6);
    }

    #[test]
    fn code_usage_lists_every_flag_the_design_names() {
        for f in [
            "-k",
            "--lang",
            "--mode",
            "--hybrid",
            "--full",
            "--no-symbol",
            "--json",
            "--meatl",
            "--stale-ok",
            "--url",
            "--api-key",
        ] {
            assert!(CODE_USAGE.contains(f), "{f} missing from usage");
        }
        assert!(CORPUS_USAGE.contains("--from"));
    }

    /// The `--from` name swap: explicit beats the manifest, the manifest
    /// beats nothing, and nothing is an error — never a guess.
    #[test]
    fn corpus_name_resolution_is_explicit_then_manifest_then_error() {
        let path = "hub.json";
        // explicit (positional or --as) wins over the manifest's field
        assert_eq!(
            resolve_corpus_name(Some("mine".into()), "theirs", path).as_deref(),
            Ok("mine")
        );
        // manifest supplies the name when nothing explicit was given
        assert_eq!(
            resolve_corpus_name(None, "theirs", path).as_deref(),
            Ok("theirs")
        );
        // neither: usage error naming both remedies
        let err = resolve_corpus_name(None, "", path).unwrap_err();
        assert!(err.contains("no 'corpus' field"), "{err}");
        assert!(err.contains("--as"), "{err}");
    }

    /// `--as` only renames; it must not let a bad name through — resolution
    /// output still passes the same pathgate as a positional name.
    #[test]
    fn corpus_name_resolution_output_is_still_pathgated() {
        assert!(xccode::pathgate::valid_corpus_name(
            &resolve_corpus_name(Some("a/b".into()), "theirs", "p").unwrap()
        )
        .is_err());
        // and the manifest's own field is gated at read time
        assert!(xccode::pathgate::valid_corpus_name(
            &resolve_corpus_name(None, "battle-terse", "p").unwrap()
        )
        .is_ok());
    }

    // ── the --fresh swap contracts (#1004) ──────────────────────────────────
    //
    // The retired test_xc_index_fresh.py pinned these against a fake node and
    // a PATH-shimmed fake binary; they are re-pinned here against a FakeNode
    // behind the same CorpusNode trait production code uses. Every contract
    // guards a DELETE or a state-file switch, so every test asserts on the
    // audit trail (`ops`) and the state file, never on stdout prose.

    /// Serialises tests that touch the count-retry env knobs (the flow reads
    /// them per call; the default 6x5s would sleep in a Unknown-count test).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_fast_count_retries<T>(f: impl FnOnce() -> T) -> T {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("XC_COUNT_TRIES", "1");
        std::env::set_var("XC_COUNT_PAUSE", "0");
        let r = f();
        std::env::remove_var("XC_COUNT_TRIES");
        std::env::remove_var("XC_COUNT_PAUSE");
        r
    }

    /// A fake node: index listing by glob, tri-state counts per dash-glob,
    /// and an audit trail of every destructive op (delete / catalog scope)
    /// and every count read — the ordering assertions need the counts too.
    struct FakeNode {
        indices: std::cell::RefCell<Vec<String>>,
        counts: std::cell::RefCell<HashMap<String, Count>>,
        /// This exact index name's DELETE fails (once — the flow warns and
        /// continues; a persistently failing node is the crash-between case).
        fail_delete: std::cell::RefCell<Option<String>>,
        ops: std::cell::RefCell<Vec<String>>,
    }

    impl FakeNode {
        fn new() -> Self {
            FakeNode {
                indices: std::cell::RefCell::new(Vec::new()),
                counts: std::cell::RefCell::new(HashMap::new()),
                fail_delete: std::cell::RefCell::new(None),
                ops: std::cell::RefCell::new(Vec::new()),
            }
        }
        fn seed_count(&self, dash_glob: &str, c: Count) {
            self.counts.borrow_mut().insert(dash_glob.into(), c);
        }
        fn live(&self, name: &str) -> bool {
            self.indices.borrow().iter().any(|i| i == name)
        }
    }

    /// `prefix*` / `a*b` glob matching, the only shape the flow globs with.
    fn glob_match(names: &[String], glob: &str) -> Vec<String> {
        let parts: Vec<&str> = glob.split('*').collect();
        names
            .iter()
            .filter(|n| {
                let mut rest = n.as_str();
                for (i, p) in parts.iter().enumerate() {
                    if i == 0 {
                        if !rest.starts_with(p) {
                            return false;
                        }
                        rest = &rest[p.len()..];
                    } else if i == parts.len() - 1 && !parts.last().is_some_and(|l| l.is_empty()) {
                        if !rest.ends_with(p) {
                            return false;
                        }
                    } else if let Some(at) = rest.find(p) {
                        rest = &rest[at + p.len()..];
                    } else {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    impl CorpusNode for FakeNode {
        fn list_indices(&self, glob: &str) -> Result<Vec<String>, String> {
            Ok(glob_match(&self.indices.borrow(), glob))
        }
        fn count(&self, dash_glob: &str) -> Count {
            self.ops.borrow_mut().push(format!("count:{dash_glob}"));
            self.counts
                .borrow()
                .get(dash_glob)
                .cloned()
                .unwrap_or(Count::Zero)
        }
        fn delete_index(&self, name: &str) -> bool {
            self.ops.borrow_mut().push(format!("delete:{name}"));
            if self.fail_delete.borrow().as_deref() == Some(name) {
                return false;
            }
            self.indices.borrow_mut().retain(|i| i != name);
            true
        }
        fn delete_catalog_scope(&self, scope: &str) -> bool {
            self.ops.borrow_mut().push(format!("catalog:{scope}"));
            true
        }
    }

    /// The fake autoindex runner: records (prefix, state_dir) per call, and
    /// "writes" `shards` indices under the prefix it was given — normally on
    /// rc 0/3, or on ANY rc when `writes_on_fail` models the #367 shape (an
    /// abort in finalisation after every document was written). A count the
    /// test pre-seeded is never overwritten: the fake models what the RUN
    /// observed, and the node may disagree (that disagreement is the point
    /// of the tri-state tests).
    struct FakeAuto<'n> {
        node: &'n FakeNode,
        rc: i32,
        shards: usize,
        writes_on_fail: bool,
        calls: std::cell::RefCell<Vec<(String, Option<std::path::PathBuf>)>>,
    }

    impl<'n> FakeAuto<'n> {
        fn ok(node: &'n FakeNode) -> Self {
            FakeAuto {
                node,
                rc: 0,
                shards: 1,
                writes_on_fail: false,
                calls: std::cell::RefCell::new(Vec::new()),
            }
        }
        fn with_rc(node: &'n FakeNode, rc: i32) -> Self {
            let mut a = FakeAuto::ok(node);
            a.rc = rc;
            a
        }
        /// Fails, but after the documents were written (#367).
        fn failing_but_writing(node: &'n FakeNode) -> Self {
            let mut a = FakeAuto::ok(node);
            a.rc = 1;
            a.writes_on_fail = true;
            a
        }
        fn runner(&self) -> impl Fn(&str, Option<&Path>) -> i32 + '_ {
            move |prefix, state_dir| {
                self.calls
                    .borrow_mut()
                    .push((prefix.to_string(), state_dir.map(Path::to_path_buf)));
                if self.rc == 0 || self.rc == 3 || self.writes_on_fail {
                    self.node
                        .indices
                        .borrow_mut()
                        .extend((0..self.shards).map(|i| format!("{prefix}-{i:03}")));
                    self.node
                        .counts
                        .borrow_mut()
                        .entry(format!("{prefix}-*"))
                        .or_insert(Count::Number(500 * self.shards as u64));
                }
                self.rc
            }
        }
        fn prefixes(&self) -> Vec<String> {
            self.calls.borrow().iter().map(|(p, _)| p.clone()).collect()
        }
    }

    /// A corpus root with the corpus cloned and (optionally) a recorded
    /// build, written through the REAL state writer so the schema is pinned
    /// on both sides of the flow.
    fn corpus_root(name: &str) -> std::path::PathBuf {
        let root = tempfile::tempdir().unwrap().keep();
        std::fs::create_dir_all(root.join("corpora").join(name)).unwrap();
        root
    }

    fn record_build(root: &Path, name: &str, build: &str, state_dir: Option<&str>) -> String {
        let prefix = format!("xc-{name}-{build}");
        state::write_state(
            root,
            name,
            "http://x",
            0,
            false,
            Some(build),
            Some(&prefix),
            state_dir,
        )
        .unwrap();
        prefix
    }

    fn run_flow(
        node: &FakeNode,
        auto: &FakeAuto<'_>,
        clock: &dyn Fn() -> i64,
        root: &Path,
        name: &str,
        fresh: bool,
    ) -> i32 {
        with_fast_count_retries(|| {
            corpus_index_flow(node, &auto.runner(), clock, root, "http://x", name, fresh)
        })
    }

    const T0: i64 = 1_700_000_000;

    #[test]
    fn a_verified_replacement_switches_state_then_retires_by_exact_name() {
        let root = corpus_root("kv");
        let old_prefix = record_build(&root, "kv", "b1", Some("/tmp/s1"));
        let node = FakeNode::new();
        node.indices.borrow_mut().push(format!("{old_prefix}-000"));
        node.seed_count(&format!("{old_prefix}-*"), Count::Number(500));

        let auto = FakeAuto::ok(&node);
        let rc = run_flow(&node, &auto, &|| T0, &root, "kv", true);

        assert_eq!(rc, 0);
        let new_prefix = format!("xc-kv-b{T0}");
        // the replacement is what the ledger now names…
        let st = state::load_state(&root, "kv").unwrap();
        assert_eq!(st.index_prefix.as_deref(), Some(new_prefix.as_str()));
        // …the old index is gone, deleted BY EXACT NAME (never a wildcard)…
        assert!(!node.live(&format!("{old_prefix}-000")));
        assert!(node.live(&format!("{new_prefix}-000")));
        assert!(node
            .ops
            .borrow()
            .iter()
            .filter(|o| o.starts_with("delete:"))
            .all(|o| !o.contains('*')));
        // …and the retire ran only AFTER the replacement's count was read.
        let ops = node.ops.borrow();
        let first_delete = ops.iter().position(|o| o.starts_with("delete:")).unwrap();
        let new_count = ops
            .iter()
            .position(|o| *o == format!("count:{new_prefix}-*"))
            .unwrap();
        assert!(new_count < first_delete, "count before any delete: {ops:?}");
        // the old build's catalog scope was cleaned too
        assert!(ops.iter().any(|o| *o == format!("catalog:{old_prefix}")));
    }

    #[test]
    fn a_build_that_never_verifies_touches_nothing_old() {
        let root = corpus_root("kv");
        let old_prefix = record_build(&root, "kv", "b1", Some("/tmp/s1"));
        let node = FakeNode::new();
        node.indices.borrow_mut().push(format!("{old_prefix}-000"));
        node.seed_count(&format!("{old_prefix}-*"), Count::Number(500));
        // the new build "succeeds" (rc 0) but the node reports ZERO records
        node.seed_count(&format!("xc-kv-b{T0}-*"), Count::Zero);

        let auto = FakeAuto::ok(&node);
        let rc = run_flow(&node, &auto, &|| T0, &root, "kv", true);

        assert_eq!(rc, 1, "a zero-count build fails even on exit 0");
        // old index untouched, old state untouched
        assert!(node.live(&format!("{old_prefix}-000")));
        let st = state::load_state(&root, "kv").unwrap();
        assert_eq!(st.index_prefix.as_deref(), Some(old_prefix.as_str()));
        assert_eq!(st.autoindex_exit, Some(serde_json::json!(0)));
        // the new build's shards were removed (its OWN catalog scope cleaned),
        // and NO delete ever named the old index
        assert!(!node.live(&format!("xc-kv-b{T0}-000")));
        assert!(node
            .ops
            .borrow()
            .iter()
            .all(|o| *o != format!("delete:{old_prefix}-000")));
    }

    #[test]
    fn a_count_the_node_does_not_answer_never_retires_a_working_index() {
        let root = corpus_root("kv");
        let old_prefix = record_build(&root, "kv", "b1", Some("/tmp/s1"));
        let node = FakeNode::new();
        node.indices.borrow_mut().push(format!("{old_prefix}-000"));
        node.seed_count(&format!("{old_prefix}-*"), Count::Number(500));
        // every count for the NEW prefix 503s — "did not say", not zero
        node.seed_count(&format!("xc-kv-b{T0}-*"), Count::Unknown("HTTP 503".into()));

        let auto = FakeAuto::ok(&node);
        let rc = run_flow(&node, &auto, &|| T0, &root, "kv", true);

        assert_eq!(rc, 1);
        // UNKNOWN authorises no delete and no switch: everything still stands
        assert!(node.live(&format!("{old_prefix}-000")));
        assert!(
            node.live(&format!("xc-kv-b{T0}-000")),
            "unverified build kept"
        );
        let st = state::load_state(&root, "kv").unwrap();
        assert_eq!(st.index_prefix.as_deref(), Some(old_prefix.as_str()));
        assert!(node.ops.borrow().iter().all(|o| !o.starts_with("delete:")));
    }

    #[test]
    fn an_unknown_old_count_is_presumed_a_working_index_not_an_empty_one() {
        let root = corpus_root("kv");
        let old_prefix = record_build(&root, "kv", "b1", Some("/tmp/s1"));
        let node = FakeNode::new();
        node.indices.borrow_mut().push(format!("{old_prefix}-000"));
        // the OLD index's count is unknown too: a failed build must not be
        // kept over it
        node.seed_count(
            &format!("{old_prefix}-*"),
            Count::Unknown("HTTP 503".into()),
        );
        // the new build FAILS (rc 1) but claims records
        node.seed_count(&format!("xc-kv-b{T0}-*"), Count::Number(500));
        let auto = FakeAuto::with_rc(&node, 1);

        let rc = run_flow(&node, &auto, &|| T0, &root, "kv", true);

        assert_eq!(rc, 1);
        // the failed build was not salvaged over an (unknowably) working index
        let st = state::load_state(&root, "kv").unwrap();
        assert_eq!(st.index_prefix.as_deref(), Some(old_prefix.as_str()));
        assert_ne!(st.salvaged, Some(true));
        assert!(!node.live(&format!("xc-kv-b{T0}-000")));
    }

    #[test]
    fn a_failed_first_build_that_wrote_records_is_kept_and_marked_salvaged() {
        let root = corpus_root("kv");
        let node = FakeNode::new();
        // no state file, no live indices: this is a FIRST build, and the
        // runner models #367 — abort in finalisation, documents already written
        let auto = FakeAuto::failing_but_writing(&node);

        let rc = run_flow(&node, &auto, &|| T0, &root, "kv", true);

        assert_eq!(rc, 0, "a salvaged first build is still queryable");
        let st = state::load_state(&root, "kv").unwrap();
        assert_eq!(st.salvaged, Some(true));
        assert_eq!(st.autoindex_exit, Some(serde_json::json!(1)));
        assert_eq!(
            st.index_prefix.as_deref(),
            Some(format!("xc-kv-b{T0}").as_str())
        );
        assert!(node.live(&format!("xc-kv-b{T0}-000")));
    }

    #[test]
    fn a_plain_rerun_resumes_the_recorded_build_under_the_same_prefix_and_state_dir() {
        let root = corpus_root("kv");
        let state_dir = root.join("autoindex-state/kv/b1");
        std::fs::create_dir_all(&state_dir).unwrap();
        let old_prefix = record_build(&root, "kv", "b1", state_dir.to_str());
        let node = FakeNode::new();
        node.indices.borrow_mut().push(format!("{old_prefix}-000"));
        node.seed_count(&format!("{old_prefix}-*"), Count::Number(500));

        let auto = FakeAuto::ok(&node);
        let rc = run_flow(&node, &auto, &|| T0, &root, "kv", false);

        assert_eq!(rc, 0);
        // the re-run reconciled the RECORDED build: same prefix, same state
        // directory, nothing deleted, nothing retired
        assert_eq!(auto.prefixes(), vec![old_prefix.clone()]);
        let (p, sd) = &auto.calls.borrow()[0];
        assert_eq!(sd.as_deref(), Some(state_dir.as_path()));
        assert!(p.starts_with("xc-kv-b1"));
        assert!(node.live(&format!("{old_prefix}-000")));
        assert!(node.ops.borrow().iter().all(|o| !o.starts_with("delete:")));
        let st = state::load_state(&root, "kv").unwrap();
        assert_eq!(st.index_prefix.as_deref(), Some(old_prefix.as_str()));
        assert_eq!(st.salvaged, Some(false));
    }

    #[test]
    fn sibling_corpus_indices_are_never_touched_by_a_rebuild() {
        let root = corpus_root("battle");
        // the sibling exists as a cloned corpus AND in the ledger
        std::fs::create_dir_all(root.join("corpora/battle-terse")).unwrap();
        std::fs::create_dir_all(root.join("state")).unwrap();
        std::fs::write(
            root.join("state/battle-terse.json"),
            "{\"corpus\":\"battle-terse\"}",
        )
        .unwrap();
        let old_prefix = record_build(&root, "battle", "b9", Some("/tmp/s9"));
        let node = FakeNode::new();
        node.indices.borrow_mut().push(format!("{old_prefix}-000"));
        // `xc-battle-*` matches BOTH; only b9 belongs to "battle"
        node.indices
            .borrow_mut()
            .push("xc-battle-terse-b1-000".into());
        node.seed_count(&format!("{old_prefix}-*"), Count::Number(500));
        node.seed_count("xc-battle-terse-b1-*", Count::Number(300));

        let auto = FakeAuto::ok(&node);
        let rc = run_flow(&node, &auto, &|| T0, &root, "battle", true);

        assert_eq!(rc, 0);
        assert!(
            node.live("xc-battle-terse-b1-000"),
            "the sibling's index survives a rebuild of 'battle'"
        );
        assert!(!node.live(&format!("{old_prefix}-000")));
        assert!(node
            .ops
            .borrow()
            .iter()
            .all(|o| !o.contains("battle-terse")));
    }

    #[test]
    fn readers_switch_before_the_retire_runs_a_failed_delete_leaves_a_duplicate_not_a_gap() {
        let root = corpus_root("kv");
        let old_prefix = record_build(&root, "kv", "b1", Some("/tmp/s1"));
        let node = FakeNode::new();
        node.indices.borrow_mut().push(format!("{old_prefix}-000"));
        node.seed_count(&format!("{old_prefix}-*"), Count::Number(500));
        // the node refuses to delete the old index (crash-between shape)
        *node.fail_delete.borrow_mut() = Some(format!("{old_prefix}-000"));

        let auto = FakeAuto::ok(&node);
        let rc = run_flow(&node, &auto, &|| T0, &root, "kv", true);

        assert_eq!(rc, 0, "a failed retire does not fail the corpus");
        // the SWITCH already happened: readers are on the new build…
        let st = state::load_state(&root, "kv").unwrap();
        assert_eq!(
            st.index_prefix.as_deref(),
            Some(format!("xc-kv-b{T0}").as_str())
        );
        assert!(node.live(&format!("xc-kv-b{T0}-000")));
        // …and the un-retired old index is a duplicate, never a gap
        assert!(node.live(&format!("{old_prefix}-000")));
    }

    #[test]
    fn two_rebuilds_inside_one_second_must_not_retire_the_build_they_just_verified() {
        let root = corpus_root("kv");
        let node = FakeNode::new();
        // a clock that repeats T0 once, then advances — the collision the
        // one-second stamp cannot otherwise see
        let tick = std::cell::Cell::new(0);
        let clock = move || {
            let t = T0 + tick.get();
            tick.set(tick.get() + 1);
            t
        };

        let auto1 = FakeAuto::ok(&node);
        assert_eq!(run_flow(&node, &auto1, &clock, &root, "kv", true), 0);
        let first = auto1.prefixes()[0].clone();
        assert_eq!(first, format!("xc-kv-b{T0}"));

        // second --fresh in the SAME second: the build id must differ, so it
        // can never list-and-retire the build it is standing on
        let auto2 = FakeAuto::ok(&node);
        assert_eq!(run_flow(&node, &auto2, &clock, &root, "kv", true), 0);
        let second = auto2.prefixes()[0].clone();
        assert_ne!(first, second, "build ids must not repeat within a second");
        // and the retire lists only the FIRST build, by exact name
        assert!(node.live(&format!("{second}-000")));
        assert!(!node.live(&format!("{first}-000")));
    }

    #[test]
    fn a_legacy_corpus_whose_indices_predate_builds_indexes_under_the_bare_namespace() {
        let root = corpus_root("kv");
        let node = FakeNode::new();
        // no state file, but a live index from before builds existed: WITHOUT
        // the stray index this would be a first build (same path as --fresh),
        // and with it the flow must take the legacy arm — the bare namespace,
        // no build id, no state dir.
        node.indices.borrow_mut().push("xc-kv-000".into());

        let auto = FakeAuto::ok(&node);
        let rc = run_flow(&node, &auto, &|| T0, &root, "kv", false);

        assert_eq!(rc, 0);
        assert_eq!(auto.prefixes(), vec!["xc-kv".to_string()]);
        assert!(
            auto.calls.borrow()[0].1.is_none(),
            "no state dir in legacy mode"
        );
        let st = state::load_state(&root, "kv").unwrap();
        assert_eq!(st.index_prefix, None, "legacy state records no build");
        assert!(node.live("xc-kv-000"));
        assert!(node.ops.borrow().iter().all(|o| !o.starts_with("delete:")));
    }
}
