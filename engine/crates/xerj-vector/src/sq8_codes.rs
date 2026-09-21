//! Ingest-time, slot-addressed SQ8 code store — issue #392.
//!
//! [`Sq8CodeStore`] is the durable counterpart of the per-query codec
//! ([`Sq8Params`](crate::quantizer::Sq8Params)): one u8 code per dimension
//! per document, written when the document is indexed and rewritten when it
//! is updated, in a single flat `Vec<u8>` addressed by a dense slot —
//! `codes[slot * dim .. slot * dim + dim]` — the same slot discipline the
//! HNSW slab uses for its f32 vectors. The codebook is fitted from the
//! vectors as they are ingested and lives with the data, so a document's
//! quantized score is a function of the index state alone and not of the
//! candidate set a particular query happens to scan (#392's second half).
//!
//! Reference points (retrieved, not copied):
//!  * qdrant `lib/quantization/src/encoded_vectors_u8.rs` (Apache-2.0) —
//!    quantized codes are written once at build time into contiguous
//!    storage, fetched by `get_vector_data(offset)` at
//!    `offset * quantized_vector_size`; the codebook (alpha/offset) is
//!    fitted at build time and persisted next to the codes; the query is
//!    encoded once and scored against stored codes without touching f32.
//!  * Lucene `OffHeapScalarQuantizedVectorValues.vectorValue(targetOrd)`
//!    seeks `targetOrd * byteSize` into a per-segment slice written at index
//!    time; requantization happens at merge time, never at query time
//!    (cited from issue #392; Elasticsearch/Lucene is AGPL/SSPL — approach
//!    only).
//!
//! The one structural difference from both: XERJ maintains the store
//! incrementally per document instead of rebuilding it per segment merge, so
//! the codebook WIDENS whenever an ingested vector falls outside the fitted
//! range (and narrows when the document holding a per-dimension bound is
//! replaced or removed), and every live code is re-encoded FROM ITS RETAINED
//! ORIGINAL VECTOR on each such re-fit. Re-encoding from the original (not
//! through a decode/encode round trip of the stored bytes — each round trip
//! drifts by up to the quantization step, and they compound) is what keeps
//! the codes bit-identical to a one-shot `Sq8Params::fit_borrowed` +
//! `encode_into` over the live set, which is the arithmetic the exact
//! scan's per-query codec performs; that bit-identity is pinned by
//! `codes_are_bit_identical_to_a_one_shot_fit_over_the_live_set` below and
//! by the engine's exact-scan honesty tests. The price is honesty about
//! memory: beside the 1 byte/dim codes the store retains the 4 bytes/dim
//! normalized f32 originals (`originals_bytes`), touched only at ingest and
//! re-fit — never on the scoring path. Widening at ingest is the same order
//! of accuracy work a merge-time requantization does, and it is what makes
//! clamping structurally impossible: the codebook always spans every live
//! vector (#371's defect class cannot recur — a vector written outside the
//! range extends the range rather than being clamped into it).

use std::collections::HashMap;

use crate::quantizer::Sq8Params;

/// Slot-addressed SQ8 code store for ONE dense_vector field.
///
/// All mutating operations are `&mut self`; the engine keeps each store
/// behind a lock and serves scans under a read guard, so a scan observes the
/// codes and the codebook of one consistent generation.
#[derive(Debug)]
pub struct Sq8CodeStore {
    dim: usize,
    /// The live codebook. `maxs` are kept alongside so an out-of-range
    /// ingest can extend the range exactly rather than through the
    /// dequantized grid.
    mins: Vec<f32>,
    maxs: Vec<f32>,
    /// slot → doc id. Slots of tombstoned documents keep their last id
    /// (never read again; `slot_of` is the live set).
    ids: Vec<String>,
    /// doc id → slot (the live set).
    slot_of: HashMap<String, u32>,
    /// Flat code array, `codes[slot * dim .. slot * dim + dim]`.
    codes: Vec<u8>,
    /// Flat ORIGINAL vector array, `originals[slot * dim .. slot * dim +
    /// dim]` — the exact (normalized) f32 each slot was encoded from,
    /// append-only like `codes`. A codebook re-fit re-encodes every live
    /// slot from these, never through its own stored bytes: a decode/encode
    /// round trip moves a value by up to the (new, wider) quantization step
    /// and the drift compounds over successive widenings, which broke
    /// bit-identity with the exact scan's per-query codec. Never read on
    /// the scoring path.
    originals: Vec<f32>,
    /// Soft delete marks, one per slot (the slab's discipline: append-only,
    /// tombstone on delete, never physically free).
    tomb: Vec<bool>,
    /// Coverage-gate denominator: number of DISTINCT doc ids ever offered to
    /// [`Self::upsert`] and not since removed. Mirrors the engine's
    /// `vector_doc_count` (counted on attempt): a rejected vector (wrong
    /// dimension) increments this without creating a slot, so
    /// `live_len() != expected` and consumers fall back to exact paths.
    expected: u64,
    /// Number of codebook widenings (observability).
    refits: u64,
}

impl Sq8CodeStore {
    /// An empty store for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Sq8CodeStore {
            dim,
            mins: vec![0.0; dim],
            maxs: vec![0.0; dim],
            ids: Vec::new(),
            slot_of: HashMap::new(),
            codes: Vec::new(),
            originals: Vec::new(),
            tomb: Vec::new(),
            expected: 0,
            refits: 0,
        }
    }

    /// Vector dimensionality this store encodes.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The current codebook (mins/scales), for decoding at query time.
    pub fn codebook(&self) -> Sq8Params {
        let scales = self
            .mins
            .iter()
            .zip(self.maxs.iter())
            .map(|(&mn, &mx)| mx - mn)
            .collect();
        Sq8Params {
            mins: self.mins.clone(),
            scales,
        }
    }

    /// Live documents holding a code slot (the coverage-gate numerator).
    pub fn live_len(&self) -> u64 {
        self.slot_of.len() as u64
    }

    /// Distinct doc ids ever accepted for this field minus removals (the
    /// coverage-gate denominator). See the field docs for the attempt
    /// semantics.
    pub fn expected(&self) -> u64 {
        self.expected
    }

    /// The serving gate: every doc id that ever carried (or attempted to
    /// carry) this field holds a live slot. Consumers that score from the
    /// codes must fall back to an exact path when this is false — a doc with
    /// a rejected vector would otherwise be silently missing from results.
    pub fn coverage_ok(&self) -> bool {
        self.live_len() == self.expected
    }

    /// Codebook widenings since construction (observability).
    pub fn refits(&self) -> u64 {
        self.refits
    }

    /// Bytes occupied by the flat code array (including tombstoned slots —
    /// the array is append-only).
    pub fn codes_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Bytes occupied by the retained original (normalized) f32 vectors —
    /// 4 bytes/dim/slot beside the 1 byte/dim codes. They exist so a
    /// codebook re-fit re-encodes from the true vectors instead of drifting
    /// through decode/encode round trips (see the module docs); they are
    /// never read on the scoring path.
    pub fn originals_bytes(&self) -> usize {
        self.originals.len() * std::mem::size_of::<f32>()
    }

    /// The live codes of one document, `dim` bytes at `slot * dim`.
    pub fn codes_for(&self, doc_id: &str) -> Option<&[u8]> {
        let &slot = self.slot_of.get(doc_id)?;
        Some(&self.codes[slot as usize * self.dim..][..self.dim])
    }

    /// Iterate every LIVE `(doc id, codes)` pair. Slot-addressed order.
    pub fn iter_live(&self) -> impl Iterator<Item = (&str, &[u8])> {
        let dim = self.dim;
        self.slot_of
            .iter()
            .map(move |(id, &slot)| (id.as_str(), &self.codes[slot as usize * dim..][..dim]))
    }

    /// Write (or rewrite, on update) one document's codes.
    ///
    /// Returns `false` when `vector` does not have exactly [`Self::dim`]
    /// dimensions — the attempt is still counted in `expected` (no slot is
    /// created), which breaks [`Self::coverage_ok`] and keeps consumers on
    /// exact paths: the same wrong-dimension document the brute-force scan
    /// skips must not be half-represented here.
    ///
    /// The invariant after every successful call: `codebook()` is the fit
    /// over the live originals and every live code is
    /// `codebook().encode(original)` — bit-identical to a one-shot
    /// `Sq8Params::fit_borrowed` + `encode_into` over the live set. A
    /// vector outside the fitted range widens the codebook; a replaced
    /// vector that held a per-dimension bound may narrow it. Either way the
    /// affected live codes are re-encoded FROM THEIR ORIGINALS (see the
    /// module docs), never through a decode/encode round trip.
    pub fn upsert(&mut self, doc_id: &str, vector: &[f32]) -> bool {
        if vector.len() != self.dim {
            // Count the attempt for a NEW id only: an id that never had a
            // slot must not leak a phantom expectation, and an id whose
            // field silently appeared with the wrong arity keeps its slot
            // (remove() is the API for that).
            if !self.slot_of.contains_key(doc_id) {
                self.expected += 1;
            }
            return false;
        }
        let live = self.slot_of.get(doc_id).copied();
        let had_live = !self.slot_of.is_empty();
        // Does this write move a per-dimension bound? Widening: the new
        // vector sits outside the fitted range. Narrowing: a REPLACED live
        // vector sat exactly on a bound, so removing it may shrink the fit
        // (a tie with another live vector keeps the bound; the re-fit that
        // follows decides, and lands back on the same codebook when nothing
        // moved).
        let mut refit = false;
        if had_live {
            for (d, &x) in vector.iter().enumerate() {
                if x < self.mins[d] || x > self.maxs[d] {
                    refit = true;
                    break;
                }
            }
            if !refit {
                if let Some(slot) = live {
                    let start = slot as usize * self.dim;
                    for d in 0..self.dim {
                        let o = self.originals[start + d];
                        if o == self.mins[d] || o == self.maxs[d] {
                            refit = true;
                            break;
                        }
                    }
                }
            }
        }
        let slot = match live {
            Some(slot) => slot,
            None => {
                let slot = self.ids.len() as u32;
                self.ids.push(doc_id.to_string());
                self.codes.extend(vec![0u8; self.dim]);
                self.originals.extend(vec![0.0f32; self.dim]);
                self.tomb.push(false);
                self.slot_of.insert(doc_id.to_string(), slot);
                self.expected += 1;
                slot
            }
        };
        let start = slot as usize * self.dim;
        self.originals[start..][..self.dim].copy_from_slice(vector);
        self.tomb[slot as usize] = false;
        if !had_live {
            // The first live vector adopts the degenerate per-dimension
            // range [v, v]: every code is 0 and decodes back to v exactly,
            // so the first document carries no quantization error.
            self.mins = vector.to_vec();
            self.maxs = vector.to_vec();
        }
        if refit {
            self.refits += 1;
            self.refit_from_originals();
        } else {
            // Even without a re-fit the slot's own codes must reflect its
            // current vector under the CURRENT codebook (an update inside
            // the range changes nothing else).
            let codebook = self.codebook();
            codebook.encode_into(vector, &mut self.codes[start..][..self.dim]);
        }
        true
    }

    /// Drop a document's live slot (delete, or the document no longer
    /// carries the field). Returns whether a slot was live. The slot is
    /// tombstoned, not freed — same append-only discipline as the HNSW
    /// slab. When the removed vector held a per-dimension bound the
    /// codebook narrows to the remaining live set and every live code is
    /// re-encoded from its original, keeping the fit equal to what a
    /// per-query codec fitted over the (now smaller) live set would
    /// compute.
    pub fn remove(&mut self, doc_id: &str) -> bool {
        match self.slot_of.remove(doc_id) {
            Some(slot) => {
                self.tomb[slot as usize] = true;
                self.expected = self.expected.saturating_sub(1);
                let start = slot as usize * self.dim;
                let mut held_bound = false;
                for d in 0..self.dim {
                    let o = self.originals[start + d];
                    if o == self.mins[d] || o == self.maxs[d] {
                        held_bound = true;
                        break;
                    }
                }
                if held_bound {
                    self.refits += 1;
                    self.refit_from_originals();
                }
                true
            }
            None => false,
        }
    }

    /// Recompute `mins`/`maxs` over the live originals and re-encode every
    /// live code from its original. After this, codes and codebook are
    /// bit-identical to `Sq8Params::fit_borrowed` over the live vectors
    /// followed by one `encode_into` per vector — the exact bytes the
    /// serving paths' per-query fallback produces, which is the bit-identity
    /// the exact-scan honesty tests pin.
    fn refit_from_originals(&mut self) {
        let dim = self.dim;
        if self.slot_of.is_empty() {
            self.mins = vec![0.0; dim];
            self.maxs = vec![0.0; dim];
            return;
        }
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];
        for &slot in self.slot_of.values() {
            let start = slot as usize * dim;
            for d in 0..dim {
                let x = self.originals[start + d];
                if x < mins[d] {
                    mins[d] = x;
                }
                if x > maxs[d] {
                    maxs[d] = x;
                }
            }
        }
        self.mins = mins;
        self.maxs = maxs;
        let codebook = self.codebook();
        for &slot in self.slot_of.values() {
            let start = slot as usize * dim;
            codebook.encode_into(
                &self.originals[start..][..dim],
                &mut self.codes[start..][..dim],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(vectors: &[Vec<f32>]) -> Sq8CodeStore {
        let mut s = Sq8CodeStore::new(vectors[0].len());
        for (i, v) in vectors.iter().enumerate() {
            assert!(s.upsert(&i.to_string(), v), "upsert {i} must succeed");
        }
        s
    }

    #[test]
    fn first_vector_encodes_exactly() {
        // Degenerate range [v, v] per dim → code 0 → decode back to v.
        let v = vec![0.25, -0.5, 1.75];
        let s = store_with(std::slice::from_ref(&v));
        let codebook = s.codebook();
        let codes = s.codes_for("0").unwrap();
        assert_eq!(codes, vec![0, 0, 0]);
        let decoded = codebook.decode(codes);
        assert_eq!(decoded, v);
    }

    #[test]
    fn roundtrip_within_one_step() {
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                (0..4)
                    .map(|d| ((i * 37 + d * 101) % 199) as f32 / 99.0 - 1.0)
                    .collect()
            })
            .collect();
        let s = store_with(&vectors);
        let codebook = s.codebook();
        for (i, v) in vectors.iter().enumerate() {
            let decoded = codebook.decode(s.codes_for(&i.to_string()).unwrap());
            let step: f32 = codebook.scales.iter().cloned().sum();
            let err: f32 = v.iter().zip(decoded).map(|(a, b)| (a - b).abs()).sum();
            assert!(
                err < step,
                "doc {i}: |err| {err} must be < one total step {step}"
            );
        }
    }

    #[test]
    fn out_of_range_vector_extends_rather_than_clamps() {
        // #371's defect class: a corpus whose dim 0 never leaves +1.0, then a
        // document rewritten to -1.0. A clamping (stale) codebook decodes
        // -1.0 back to +1.0; a widening one decodes it near -1.0.
        let mut s = store_with(&[vec![1.0, 0.0], vec![1.0, 0.5]]);
        let before_refits = s.refits();
        assert!(s.upsert("0", &[-1.0, 0.0]));
        assert_eq!(s.refits(), before_refits + 1, "must have widened");
        let decoded = s.codebook().decode(s.codes_for("0").unwrap());
        assert!(
            (decoded[0] - (-1.0)).abs() < 2.0 / 255.0,
            "dim 0 decoded to {} — the codebook clamped (#371 defect class)",
            decoded[0]
        );
        // The untouched dimension-1 values of the OTHER document survive the
        // re-encode within a step.
        let d1 = s.codebook().decode(s.codes_for("1").unwrap());
        assert!((d1[0] - 1.0).abs() < 2.0 / 255.0);
        assert!((d1[1] - 0.5).abs() < 2.0 / 255.0);
    }

    #[test]
    fn update_rewrites_the_slot() {
        let mut s = store_with(&[vec![0.0, 1.0], vec![1.0, 0.0]]);
        let codebook_before = s.codebook();
        let old = codebook_before.decode(s.codes_for("0").unwrap());
        assert_eq!(old, vec![0.0, 1.0]);
        assert!(s.upsert("0", &[1.0, 1.0]));
        // Both dims' ranges now [0,1]; doc 0 re-encoded in place.
        assert_eq!(s.live_len(), 2, "update must not grow the live set");
        let decoded = s.codebook().decode(s.codes_for("0").unwrap());
        assert!((decoded[0] - 1.0).abs() < 1.0 / 255.0);
        assert!((decoded[1] - 1.0).abs() < 1.0 / 255.0);
    }

    #[test]
    fn remove_breaks_coverage_and_delete_reinsert_restores_it() {
        let mut s = store_with(&[vec![0.0], vec![1.0]]);
        assert!(s.coverage_ok());
        assert!(s.remove("0"));
        assert_eq!(s.live_len(), 1);
        assert!(s.coverage_ok(), "remove decrements both sides");
        assert!(!s.remove("0"), "second remove is a no-op");
        assert!(s.upsert("0", &[0.5]));
        assert!(s.coverage_ok());
        assert_eq!(s.live_len(), 2);
    }

    #[test]
    fn wrong_dim_attempt_breaks_coverage() {
        let mut s = store_with(&[vec![0.0, 0.0]]);
        assert!(!s.upsert("bad", &[0.0, 0.0, 0.0]), "mismatch rejected");
        assert_eq!(s.live_len(), 1);
        assert_eq!(s.expected(), 2);
        assert!(
            !s.coverage_ok(),
            "a rejected vector must keep consumers on exact paths"
        );
        assert!(s.codes_for("bad").is_none());
    }

    #[test]
    fn codes_bytes_is_one_byte_per_dim_per_slot() {
        let s = store_with(&[vec![0.0; 8], vec![1.0; 8], vec![0.5; 8]]);
        assert_eq!(s.codes_bytes(), 3 * 8);
    }

    #[test]
    fn iter_live_skips_removed_and_addresses_by_slot() {
        let mut s = store_with(&[vec![0.0, 0.1], vec![0.2, 0.3]]);
        assert!(s.remove("0"));
        let live: Vec<&str> = s.iter_live().map(|(id, _)| id).collect();
        assert_eq!(live, vec!["1"]);
        let (_, codes) = s.iter_live().next().unwrap();
        assert_eq!(codes.len(), 2);
    }

    #[test]
    fn codebook_is_order_independent() {
        // The final range spans the same values regardless of ingest order,
        // so a restart rebuild produces the same codebook — and, since every
        // re-fit re-encodes from the retained originals rather than through
        // decode/encode round trips, the same CODES bit for bit (ids are
        // positional here, so the comparison is per VECTOR).
        let a: Vec<Vec<f32>> = vec![vec![0.0, 1.0], vec![-2.0, 0.5], vec![3.0, -1.0]];
        let mut b = a.clone();
        b.reverse();
        let sa = store_with(&a);
        let sb = store_with(&b);
        assert_eq!(sa.codebook().mins, sb.codebook().mins);
        assert_eq!(sa.codebook().scales, sb.codebook().scales);
        for v in &a {
            let pos_a = a.iter().position(|x| x == v).unwrap();
            let pos_b = b.iter().position(|x| x == v).unwrap();
            assert_eq!(
                sa.codes_for(&pos_a.to_string()),
                sb.codes_for(&pos_b.to_string()),
                "codes for {v:?}"
            );
        }
    }

    #[test]
    fn codes_are_bit_identical_to_a_one_shot_fit_over_the_live_set() {
        // #392's serving claim, pinned at the codec level: after ANY
        // sequence of upserts (fresh writes, in-range and out-of-range
        // updates) and removes, the store's codebook and every live code
        // are bit-identical to fitting `Sq8Params` over the live vectors
        // once and encoding each under that fit — the exact bytes the
        // per-query fallback codec computes. The decode/encode round-trip
        // re-encode this test guards against drifted by up to the
        // quantization step per widening, which broke bit-identity between
        // the codes serving path and the exact scan.
        let mut s = Sq8CodeStore::new(4);
        let mut live: Vec<(String, Vec<f32>)> = Vec::new();
        let upsert =
            |s: &mut Sq8CodeStore, live: &mut Vec<(String, Vec<f32>)>, id: &str, v: Vec<f32>| {
                s.upsert(id, &v);
                if let Some(slot) = live.iter_mut().find(|(i, _)| i == id) {
                    slot.1 = v;
                } else {
                    live.push((id.to_string(), v));
                }
            };
        let v = |a: f32, b: f32, c: f32, d: f32| vec![a, b, c, d];
        // Fresh writes (several widen the range), then an out-of-range
        // update, an in-range update, a delete of a bound holder, and a
        // delete of a non-bound doc.
        upsert(&mut s, &mut live, "a", v(0.0, 0.5, -0.5, 0.1));
        upsert(&mut s, &mut live, "b", v(1.0, -1.0, 0.5, -0.9));
        upsert(&mut s, &mut live, "c", v(-0.7, 0.2, 0.9, 0.3));
        upsert(&mut s, &mut live, "d", v(0.4, 0.4, 0.4, 0.4));
        upsert(&mut s, &mut live, "b", v(-3.0, 0.0, 2.0, 0.0)); // out-of-range update
        upsert(&mut s, &mut live, "c", v(0.1, 0.1, 0.1, 0.1)); // in-range update
        assert!(s.remove("a")); // held dim-1 max / dim-2 min before its update
        assert!(s.remove("d")); // interior doc: no bound moves
        let live_now: Vec<&[f32]> = live
            .iter()
            .filter(|(id, _)| id != "a" && id != "d")
            .map(|(_, v)| v.as_slice())
            .collect();
        let fit = Sq8Params::fit_borrowed(live_now.iter().copied(), 4);
        let book = s.codebook();
        assert_eq!(
            book.mins, fit.mins,
            "codebook mins must equal the one-shot fit"
        );
        assert_eq!(
            book.scales, fit.scales,
            "codebook scales must equal the one-shot fit"
        );
        for (id, _) in live.iter() {
            if id == "a" || id == "d" {
                continue;
            }
            assert_eq!(
                s.codes_for(id).map(<[u8]>::to_vec),
                Some(
                    fit.encode(
                        &live
                            .iter()
                            .find(|(i, _)| i == id)
                            .map(|(_, v)| v.clone())
                            .unwrap()
                    )
                ),
                "codes for {id} must be bit-identical to encode-under-fit"
            );
        }
    }
}
