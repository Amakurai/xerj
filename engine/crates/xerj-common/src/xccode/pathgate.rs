//! The hard gate every manifest-derived path passes BEFORE anything is built
//! from it.
//!
//! A hub manifest is UNTRUSTED INPUT: `--from` is documented as "rebuild a
//! corpus someone else defined", so the file arrives from a hub, a chat
//! message or a pull request, and every field in it is attacker-controlled.
//! `repo` becomes a filesystem path (`corpora/<name>/<repo>`) that the
//! lifecycle then runs `git checkout --force` and `git clean -fd` inside, so
//! a name like `../../../work/repo` walks out of ~/.xerj-code and destroys
//! uncommitted work in an unrelated checkout.
//!
//! The original script validated twice (python then bash) because of the
//! checkout; the port validates ONCE, unskippably, before any path exists.

/// Subcommand nouns a corpus name must not shadow — `xerj corpus list` sorts
/// by these words' meaning, and a corpus called "index" or "list" makes the
/// lifecycle's own help text ambiguous.
pub const RESERVED_CORPUS_NAMES: &[&str] = &[
    "code", "corpus", "add", "index", "list", "def", "search", "help",
];

/// Validate a corpus name: no `/` (no path escape), no `*` (no glob
/// injection into index patterns), no leading `.` (no hidden dir, no `..`),
/// not empty, not a reserved subcommand noun.
pub fn valid_corpus_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("corpus name must not be empty".to_string());
    }
    if name.contains('/') {
        return Err(format!(
            "bad corpus name '{name}': it becomes a directory under the corpora root, so \
             '/' would escape it"
        ));
    }
    if name.contains('*') {
        return Err(format!(
            "bad corpus name '{name}': '*' would be interpreted as an index wildcard"
        ));
    }
    if name.starts_with('.') {
        return Err(format!(
            "bad corpus name '{name}': a leading '.' hides the corpus directory"
        ));
    }
    if RESERVED_CORPUS_NAMES.contains(&name) {
        return Err(format!(
            "corpus name '{name}' is reserved — it is a subcommand of `xerj code`/`xerj corpus`"
        ));
    }
    Ok(())
}

/// Validate a repo name from an untrusted manifest: `^[A-Za-z0-9_][A-Za-z0-9._-]*$`.
///
/// The message says WHY because the rule looks arbitrary until the checkout
/// is mentioned: this name becomes a path that is then force-checked-out and
/// cleaned.
pub fn valid_repo_name(repo: &str) -> Result<(), String> {
    let mut chars = repo.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
    let rest_ok = chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if first_ok && rest_ok {
        return Ok(());
    }
    Err(format!(
        "repo name '{repo}' is not a plain directory name. It is used as a path under the \
         corpus directory and is then force-checked-out and cleaned, so a name containing \
         '/' or '..' would overwrite files outside the corpus. Allowed: letters, digits, \
         '.', '_' and '-', not starting with '.' or '-'."
    ))
}

/// Validate a pinned sha: full 40 lowercase hex.
///
/// A short sha is not fetchable from a remote (`git fetch --depth 1 origin
/// <short>` fails with "couldn't find remote ref"), so a manifest carrying
/// one would rebuild silently at the tip — the opposite of a pin. Fail
/// loudly, with the regeneration advice, instead.
pub fn valid_sha(repo: &str, sha: &str) -> Result<(), String> {
    let full_hex = sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit());
    // The detector records what git prints; lowercase is the convention, but
    // rejecting uppercase would break manifests written by tools that
    // uppercase — accept either case, both are full shas.
    if full_hex {
        Ok(())
    } else {
        Err(format!(
            "repo '{repo}' has sha '{sha}', which is not a full 40-character sha. A short \
             sha is not fetchable from a remote, so this manifest cannot pin a rebuild. \
             Regenerate it with `xerj corpus add <name> <url>...` on a machine that has \
             the corpus."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_corpus_names_are_rejected_before_any_path_is_built() {
        for bad in ["", "/", "a/b", "*", "a*b", ".", "..", ".hidden"] {
            assert!(valid_corpus_name(bad).is_err(), "{bad:?} must be rejected");
        }
        for reserved in RESERVED_CORPUS_NAMES {
            assert!(
                valid_corpus_name(reserved).is_err(),
                "'{reserved}' is a subcommand noun"
            );
        }
        for good in ["kv", "xerj-search", "battle-terse", "rust_text", "c++"] {
            assert!(valid_corpus_name(good).is_ok(), "{good:?} is a fine name");
        }
    }

    #[test]
    fn unsafe_repo_names_are_rejected_with_the_path_escape_reason() {
        for bad in [
            "../escape",
            "a/b",
            ".hidden",
            "-flag",
            "",
            "repo name",
            "$(boom)",
        ] {
            let err = valid_repo_name(bad).unwrap_err();
            assert!(
                err.contains("force-checked-out"),
                "the message must say why: {err}"
            );
        }
        for good in ["valkey", "ClickHouse", "x_2", "a.b-c", "Sled"] {
            assert!(valid_repo_name(good).is_ok(), "{good:?}");
        }
    }

    #[test]
    fn short_shas_are_rejected_with_regen_advice() {
        let err = valid_sha("valkey", "e449d17").unwrap_err();
        assert!(
            err.contains("not a full 40-character sha")
                && err.contains("Regenerate it with `xerj corpus add"),
            "{err}"
        );
        let ok = "31081d9f05014003321333553bb3e657eb3da168";
        assert!(valid_sha("ClickHouse", ok).is_ok());
        assert!(valid_sha("ClickHouse", &ok.to_uppercase()).is_ok());
        assert!(valid_sha("x", &format!("{ok}1")).is_err(), "41 chars");
        assert!(valid_sha("x", "zzz").is_err());
    }
}
