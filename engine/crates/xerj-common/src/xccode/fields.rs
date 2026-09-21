//! Mapping-resolved query fields and semantic-capable index discovery.

use serde_json::Value;

/// These exact fields and these exact (flat) weights were measured, not
/// chosen. Swept 7 variants over 6 ground-truth queries on a 324-file Rust
/// corpus (`body, defs, title` flat: top3 6/6 — the winner; boosting `defs`
/// favours test modules; `title^2 body`: top3 2/6). Full table in
/// tools/xerj-code/SKILL.md; provenance: measure/SERVER_UPLIFT_SCORECARD.md.
///
/// Two fields are deliberately absent:
/// * `symbols.name` — `symbols` is an array of objects with no searchable
///   `.name` subpath; including it makes the whole multi_match return ZERO
///   hits with no error at all (it took an Aho-Corasick query from 0 hits to
///   3 correct files just to remove it).
/// * `"*"` — a bare wildcard flattens every score to the same value
///   (measured: every hit scored exactly 2.0).
///
/// `defs_expanded^0.5` is a low-weight RECALL field (per-symbol signatures +
/// identifier sub-words) boosted BELOW 1.0 on purpose. It is NOT safe to send
/// unconditionally — see [`resolve_fields`].
pub const FIELDS: &[&str] = &["body", "defs", "title", "defs_expanded^0.5"];

/// Drop query fields that no index under the prefix actually maps.
///
/// This engine does NOT ignore an unmapped field in `multi_match` the way ES
/// does — including one silently collapses a MULTI-TOKEN query to ZERO hits.
/// Measured 2026-08-06, one index, exact totals (`relation: eq`):
///
/// ```text
/// query "log merge policy segment size buckets"
/// fields=["body"]                                 -> 673 hits
/// fields=["body","defs"]                          -> 673 hits
/// fields=["body","defs","title"]                  -> 673 hits
/// fields=["body","defs","title","defs_expanded"]  ->   0 hits
/// ```
///
/// `defs_expanded` was mapped in exactly 0 of the corpus's 219 indices, so
/// every multi-word query returned "no passage matches" — the silent-zero
/// failure shape. Full reproducer: measure/MULTIMATCH_DEFECT.md.
///
/// `mapping` is `None` when the mapping could not be read (transport hiccup,
/// auth): send the list UNCHANGED rather than silently narrowing it. The
/// floor is `["body"]` — never an empty field list.
pub fn resolve_fields(mapping: Option<&Value>) -> Vec<String> {
    let Some(mapping) = mapping else {
        return FIELDS.iter().map(|s| s.to_string()).collect();
    };
    let Some(obj) = mapping.as_object() else {
        return FIELDS.iter().map(|s| s.to_string()).collect();
    };
    let mut present = std::collections::BTreeSet::new();
    for m in obj.values() {
        if let Some(props) = m.pointer("/mappings/properties").and_then(Value::as_object) {
            for key in props.keys() {
                present.insert(key.clone());
            }
        }
    }
    let out: Vec<String> = FIELDS
        .iter()
        .filter(|f| {
            let base = f.split('^').next().unwrap_or(f);
            present.contains(base)
        })
        .map(|s| s.to_string())
        .collect();
    if out.is_empty() {
        vec!["body".to_string()]
    } else {
        out
    }
}

/// Indices under the prefix whose `field` (default `body`) is mapped as
/// `semantic_text`.
///
/// A `semantic` query against an index where the field is plain `text` does
/// not return fewer hits — it fails the WHOLE search with a 400, taking every
/// other index in the wildcard down with it. So the capable set is discovered
/// from the mapping and the vector arm is aimed ONLY at those indices.
///
/// Returns `(capable, total)`, names sorted. An unreadable/unparseable
/// mapping yields an empty capable set (degrade to BM25), never a guess.
pub fn semantic_capable(mapping: Option<&Value>) -> (Vec<String>, usize) {
    let Some(mapping) = mapping.and_then(Value::as_object) else {
        return (Vec::new(), 0);
    };
    let mut capable: Vec<String> = mapping
        .iter()
        .filter(|(_, m)| {
            m.pointer("/mappings/properties/body/type")
                .and_then(Value::as_str)
                .is_some_and(|t| t == "semantic_text")
        })
        .map(|(name, _)| name.clone())
        .collect();
    capable.sort();
    (capable, mapping.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn mapping(indices: &[(&str, &[&str], bool)]) -> Value {
        let mut obj = serde_json::Map::new();
        for (name, fields, semantic_body) in indices {
            let mut props = serde_json::Map::new();
            for f in fields.iter() {
                props.insert(
                    f.to_string(),
                    json!({ "type": if *semantic_body && *f == "body" { "semantic_text" } else { "text" } }),
                );
            }
            obj.insert(
                name.to_string(),
                json!({ "mappings": { "properties": props } }),
            );
        }
        Value::Object(obj)
    }

    #[test]
    fn unmapped_fields_are_dropped_and_the_floor_is_body() {
        // The MULTIMATCH_DEFECT guard: `defs_expanded` unmapped in every
        // index must be dropped, not sent.
        let m = mapping(&[
            ("xc-kv-b1-000", &["body", "defs", "title"], false),
            ("xc-kv-b1-001", &["body", "defs"], false),
        ]);
        assert_eq!(
            resolve_fields(Some(&m)),
            vec!["body".to_string(), "defs".into(), "title".into()]
        );

        // No FIELDS member mapped at all -> the ["body"] floor, never empty.
        let thin = mapping(&[("i", &["unrelated"], false)]);
        assert_eq!(resolve_fields(Some(&thin)), vec!["body".to_string()]);
    }

    #[test]
    fn an_unreadable_mapping_sends_the_full_list() {
        assert_eq!(
            resolve_fields(None),
            FIELDS.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
        assert_eq!(
            resolve_fields(Some(&json!("junk"))),
            FIELDS.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn capable_discovery_finds_only_semantic_text_indices() {
        let m = mapping(&[
            ("plain-1", &["body", "defs"], false),
            ("sem-1", &["body", "defs"], true),
            ("sem-2", &["body"], true),
        ]);
        let (capable, total) = semantic_capable(Some(&m));
        assert_eq!(capable, vec!["sem-1".to_string(), "sem-2".to_string()]);
        assert_eq!(total, 3);
        assert!(semantic_capable(None) == (Vec::new(), 0));
    }
}
