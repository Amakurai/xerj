//! `corpora/<corpus>/corpus.json` — the per-corpus manifest — plus hub
//! manifests (`--from`), which are UNTRUSTED INPUT and validated as such.
//!
//! The manifest is regenerated FROM DISK at clone time, never copied through
//! from the input, so it always describes the checkout it sits beside. Its
//! byte format is pinned: humans diff these files by hand and share them in
//! chats, so the shape must not churn between releases.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::{Map, Value};

use super::pathgate;

/// One repo entry. Tolerant parse: fields the writer has not always emitted
/// (`files`, `bytes`, `review`) are optional, unknown keys are ignored, so a
/// manifest written by any historical version still loads.
#[derive(Debug, Clone, Deserialize)]
pub struct ManifestRepo {
    pub repo: String,
    pub url: String,
    #[serde(default)]
    pub sha: String,
    #[serde(default)]
    pub licence: String,
    #[serde(default)]
    pub files: Option<u64>,
    #[serde(default)]
    pub bytes: Option<u64>,
    /// Vetted hub records carry a review block (`spdx`, `use`, `by`, `at`,
    /// `note`). It is PRESERVED verbatim across a rebuild when repo+sha
    /// match — a human's licence review must survive a re-index.
    #[serde(default)]
    pub review: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CorpusManifest {
    #[serde(default)]
    pub corpus: String,
    #[serde(default)]
    pub cloned_at: Option<String>,
    #[serde(default)]
    pub repos: Vec<ManifestRepo>,
}

/// Read a corpus manifest. Missing file is `Err` with the path named — the
/// caller decides whether that is fatal (`corpus add` regenerates it;
/// `xerj code` reports the corpus as licence-unmapped).
pub fn read_corpus_manifest(path: &Path) -> Result<CorpusManifest, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read manifest {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("{} is not valid JSON: {e}", path.display()))
}

/// repo -> recorded licence for a corpus. ALWAYS the recorded string, never
/// re-derived at query time: the licence decision was made once, at clone
/// time, against the checkout — re-deriving it per query is how prose drift
/// becomes load-bearing.
pub fn licence_map(root: &Path, corpus: &str) -> HashMap<String, String> {
    let path = root.join("corpora").join(corpus).join("corpus.json");
    read_corpus_manifest(&path)
        .map(|m| {
            m.repos
                .iter()
                .map(|r| (r.repo.clone(), r.licence.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// One repo entry as the writer emits it. Compact, key order pinned
/// (`repo,url,licence,sha,files,bytes[,review]`) — byte-identical to the
/// script's hand-built line so existing manifests diff clean against new ones.
fn entry_json(repo: &ManifestRepo) -> String {
    let mut m = Map::new();
    m.insert("repo".into(), Value::String(repo.repo.clone()));
    m.insert("url".into(), Value::String(repo.url.clone()));
    m.insert("licence".into(), Value::String(repo.licence.clone()));
    m.insert("sha".into(), Value::String(repo.sha.clone()));
    if let Some(f) = repo.files {
        m.insert("files".into(), Value::from(f));
    }
    if let Some(b) = repo.bytes {
        m.insert("bytes".into(), Value::from(b));
    }
    if let Some(r) = &repo.review {
        m.insert("review".into(), r.clone());
    }
    Value::Object(m).to_string()
}

/// Write the manifest in the pinned format:
///
/// ```text
/// {"corpus":"kv","cloned_at":"2026-08-06T12:00:00Z","repos":[
///   {"repo":"valkey","url":"https://github.com/valkey/valkey","licence":"BSD-3-Clause","sha":"abc…","files":912,"bytes":1234567}
/// ]}
/// ```
///
/// Atomic (tmp + rename): a half-written manifest describes a checkout that
/// does not exist, and `xerj corpus list` reads these.
pub fn write_corpus_manifest(path: &Path, corpus: &str, cloned_at: &str, repos: &[ManifestRepo]) {
    let mut body = format!(
        "{{\"corpus\":{},\"cloned_at\":{},\"repos\":[\n",
        Value::String(corpus.to_string()),
        Value::String(cloned_at.to_string())
    );
    for (i, r) in repos.iter().enumerate() {
        if i > 0 {
            body.push_str(",\n");
        }
        body.push_str("  ");
        body.push_str(&entry_json(r));
    }
    body.push_str("\n]}\n");
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, body).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// A validated row of a hub manifest, ready for the clone loop.
#[derive(Debug, Clone)]
pub struct HubRow {
    pub repo: String,
    pub url: String,
    pub sha: String,
    pub declared_licence: String,
}

/// A whole validated hub manifest: the corpus name plus its rows. The
/// `corpus` field is the name `xerj corpus add --from` uses when the
/// caller gave none — dropping it (an earlier port bug) made `--from`
/// unusable without a redundant positional.
#[derive(Debug, Clone)]
pub struct HubManifest {
    pub corpus: String,
    pub rows: Vec<HubRow>,
}

/// Parse and VALIDATE a hub manifest (`--from`). Untrusted input rules from
/// the original `read_manifest`, enforced once, here, before any path is
/// built from a field: `repo` becomes a directory that is later
/// force-checked-out and cleaned inside, and a short sha is not fetchable,
/// so it silently rebuilds at the tip — the opposite of a pin.
pub fn read_hub_manifest(path: &Path) -> Result<HubManifest, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("no such manifest: {} ({e})", path.display()))?;
    let v: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;
    let repos = v
        .get("repos")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| format!("{} has no 'repos' array", path.display()))?;
    let corpus = v
        .get("corpus")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if !corpus.is_empty() {
        pathgate::valid_corpus_name(&corpus).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    let mut out = Vec::new();
    for r in repos {
        let repo = r.get("repo").and_then(Value::as_str).unwrap_or("");
        let url = r.get("url").and_then(Value::as_str).unwrap_or("");
        if repo.is_empty() || url.is_empty() {
            return Err(format!(
                "{}: an entry is missing 'repo' or 'url'",
                path.display()
            ));
        }
        pathgate::valid_repo_name(repo).map_err(|e| format!("{}: {e}", path.display()))?;
        let sha = r.get("sha").and_then(Value::as_str).unwrap_or("");
        pathgate::valid_sha(repo, sha).map_err(|e| format!("{}: {e}", path.display()))?;
        out.push(HubRow {
            repo: repo.to_string(),
            url: url.to_string(),
            sha: sha.to_string(),
            declared_licence: r
                .get("licence")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        });
    }
    Ok(HubManifest { corpus, rows: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tmp() -> std::path::PathBuf {
        tempfile::tempdir().unwrap().keep()
    }

    #[test]
    fn round_trip_is_byte_stable_and_review_survives() {
        let dir = tmp();
        let path = dir.join("corpus.json");
        let review = json!({ "spdx": "Apache-2.0", "use": "search engine internals", "by": "adi", "at": "2026-08-06", "note": "safe" });
        let repos = vec![ManifestRepo {
            repo: "valkey".into(),
            url: "https://github.com/valkey/valkey".into(),
            licence: "BSD-3-Clause".into(),
            sha: "31081d9f05014003321333553bb3e657eb3da168".into(),
            files: Some(912),
            bytes: Some(1_234_567),
            review: Some(review.clone()),
        }];
        write_corpus_manifest(&path, "kv", "2026-08-06T12:00:00Z", &repos);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text,
            "{\"corpus\":\"kv\",\"cloned_at\":\"2026-08-06T12:00:00Z\",\"repos\":[\n  \
             {\"repo\":\"valkey\",\"url\":\"https://github.com/valkey/valkey\",\"licence\":\
             \"BSD-3-Clause\",\"sha\":\"31081d9f05014003321333553bb3e657eb3da168\",\"files\":912,\
             \"bytes\":1234567,\"review\":{\"spdx\":\"Apache-2.0\",\"use\":\"search engine \
             internals\",\"by\":\"adi\",\"at\":\"2026-08-06\",\"note\":\"safe\"}}\n]}\n"
        );
        let back = read_corpus_manifest(&path).unwrap();
        assert_eq!(back.repos[0].review, Some(review));
        assert_eq!(back.corpus, "kv");
    }

    #[test]
    fn licence_map_comes_from_the_record_not_the_prose() {
        let dir = tmp();
        let corpora = dir.join("corpora").join("xerj-search");
        std::fs::create_dir_all(&corpora).unwrap();
        write_corpus_manifest(
            &corpora.join("corpus.json"),
            "xerj-search",
            "t",
            &[
                ManifestRepo {
                    repo: "tantivy".into(),
                    url: "u1".into(),
                    licence: "Apache-2.0/MIT".into(),
                    sha: "s".into(),
                    files: None,
                    bytes: None,
                    review: None,
                },
                ManifestRepo {
                    repo: "sonic".into(),
                    url: "u2".into(),
                    licence: "MPL-2.0".into(),
                    sha: "s".into(),
                    files: None,
                    bytes: None,
                    review: None,
                },
            ],
        );
        let m = licence_map(&dir, "xerj-search");
        assert_eq!(m.get("sonic").map(String::as_str), Some("MPL-2.0"));
        assert!(!m.contains_key("nope"));
        assert!(licence_map(&dir, "never-cloned").is_empty());
    }

    #[test]
    fn hub_validation_rejects_path_escapes_and_short_shas() {
        let dir = tmp();
        let bad = dir.join("evil.json");
        std::fs::write(&bad, json!({ "repos": [ { "repo": "../../work", "url": "u", "sha": "31081d9f05014003321333553bb3e657eb3da168" } ] }).to_string()).unwrap();
        let err = read_hub_manifest(&bad).unwrap_err();
        assert!(err.contains("force-checked-out"), "{err}");

        std::fs::write(
            &bad,
            json!({ "repos": [ { "repo": "valkey", "url": "u", "sha": "e449d17" } ] }).to_string(),
        )
        .unwrap();
        let err = read_hub_manifest(&bad).unwrap_err();
        assert!(err.contains("not a full 40-character sha"), "{err}");

        std::fs::write(
            &bad,
            json!({ "repos": [ { "repo": "valkey" } ] }).to_string(),
        )
        .unwrap();
        assert!(read_hub_manifest(&bad)
            .unwrap_err()
            .contains("missing 'repo' or 'url'"));
        std::fs::write(&bad, json!({ "repos": [] }).to_string()).unwrap();
        assert!(read_hub_manifest(&bad)
            .unwrap_err()
            .contains("no 'repos' array"));
    }

    /// The hub's own vetted manifests must stay valid under the SAME gate the
    /// `--from` path enforces — they are the reference for "rebuild a corpus
    /// someone else defined", so a hub file that the tool itself would reject
    /// is a bug in the hub. Reads them from the repo via CARGO_MANIFEST_DIR
    /// ancestors (the published_schema_drift.rs house pattern).
    #[test]
    fn hub_manifests_pass_the_untrusted_input_gate() {
        let hub = find_repo_root().join("tools/xerj-code/hub");
        let mut checked = 0;
        for entry in std::fs::read_dir(&hub).expect("hub dir") {
            let p = entry.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let hub = read_hub_manifest(&p)
                .unwrap_or_else(|e| panic!("{} failed the gate: {e}", p.display()));
            assert!(!hub.rows.is_empty());
            // Every hub manifest names its corpus — `--from` depends on it.
            assert!(
                !hub.corpus.is_empty(),
                "{} has no 'corpus' field",
                p.display()
            );
            checked += 1;
        }
        assert!(
            checked >= 1,
            "no hub manifests found under {}",
            hub.display()
        );
    }

    fn find_repo_root() -> std::path::PathBuf {
        let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            if dir.join("tools/xerj-code/hub").is_dir() {
                return dir;
            }
            if !dir.pop() {
                panic!("repo root not found above {}", env!("CARGO_MANIFEST_DIR"));
            }
        }
    }
}
