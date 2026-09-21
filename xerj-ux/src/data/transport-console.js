// ============================================================
// XERJ Console — reader transport for the signed-in operator
//
// A console session is NOT an engine credential. Everything the reader can
// get, it gets through the session-authenticated console API, so it works
// the same on an auth-enabled engine (the default) as on `--insecure`:
//
//   search   the panel proxy (one exact index; the proxy refuses patterns,
//            system indices and the reserved brain namespace — see
//            xerj-console-api/src/data_sources.rs).
//   mapping  the console's field list for the index, reshaped to the
//            `_mapping` wire form data/reader-api.js reads.
//   ego      the console's graph read path
//            (`/_xerj-console/api/v1/graph/{brain}/ego`) — session-authorized
//            for the operator roles (owner/admin), same §4.3 contract as the
//            data plane (xerj-console-api/src/graph.rs, issue #936).
//   discoverBrains
//            one call to the console's brains listing
//            (`/_xerj-console/api/v1/graph/brains`), which returns each
//            brain's `nodes_index` — this replaces the old direct
//            `_cat/indices/.xerj-memory-*` walk plus per-brain meta-doc
//            reads, both of which 401 for a session.
//
// A role that may not read brains (editor/viewer) gets the same 404 an
// unknown brain gets, so the reader says "no brain for this index" — the
// honest user-facing statement for that role — and never a leak.
// ============================================================

import { fieldTypes } from './console-index-api.js';

const PROXY = '/_xerj-console/api/v1/data-sources/connections/built-in/indices';
const GRAPH = '/_xerj-console/api/v1/graph';
/** How many of the discovered brains a record's graph panel walks. Two
 *  `xerj brain` runs over different folders land in the same `ax-docs`
 *  index with two brains that both list it. */
export const BRAINS_WALKED = 4;
const enc = encodeURIComponent;

function httpError(status) {
  const e = new Error(`HTTP ${status}`);
  e.status = status;
  e.kind = status === 401 ? 'unauthorized' : status === 403 ? 'forbidden' : status === 404 ? 'not-found' : 'http';
  return e;
}

/**
 * `fetch`, with a network failure given a KIND. A browser reports "the engine
 * is not there" as a bare `TypeError: Failed to fetch`; reader-api.js only
 * translates errors that carry `.kind`, so the operator read the raw browser
 * string where a guest read "engine unreachable" (PR #945 review).
 */
async function request(url, init) {
  try {
    return await fetch(url, init);
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    const err = new Error('engine unreachable');
    err.kind = 'network';
    throw err;
  }
}

export function makeConsoleTransport() {
  return {
    guest: false,
    async search(index, body, signal) {
      const r = await request(`${PROXY}/${enc(index)}/search`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify(body),
        signal,
      });
      if (!r.ok) throw httpError(r.status);
      return r.json(); // the proxy answers in ES `_search` wire shape, unwrapped
    },
    async mapping(index, signal) {
      const types = await fieldTypes(index, signal);
      const properties = {};
      for (const [name, type] of Object.entries(types)) properties[name] = { type };
      return { [index]: { mappings: { properties } } };
    },
    async ego(brain, params, signal) {
      const qs = new URLSearchParams(params);
      const r = await request(`${GRAPH}/${enc(brain)}/ego?${qs}`, { signal, credentials: 'same-origin', headers: { accept: 'application/json' } });
      let body = null;
      try { body = await r.json(); } catch { body = null; }
      return { status: r.status, body };
    },
    /**
     * EVERY brain whose meta doc lists `index` in `nodes_index` (what
     * `xerj brain` writes), in listing order, at most BRAINS_WALKED of them.
     * `[]` when none does or nothing can be read. Best-effort; never throws.
     *
     * A list, not the first match: after `xerj brain casefile` and
     * `xerj brain vendors` on one node both brains list `ax-docs`, and each
     * holds links only for its own files. Returning the first brain made the
     * reader say "records no links for this record" for half the corpus,
     * with which half depending on listing order (PR #945 review).
     */
    async discoverBrains(index, signal) {
      const out = [];
      try {
        const r = await fetch(`${GRAPH}/brains`, { signal, credentials: 'same-origin', headers: { accept: 'application/json' } });
        if (!r.ok) return out;
        const j = await r.json();
        for (const b of (j && j.data && j.data.brains) || []) {
          const ni = String((b && b.nodes_index) || '');
          if (ni.split(',').map((x) => x.trim()).includes(index)) out.push(b.name);
          if (out.length >= BRAINS_WALKED) break;
        }
      } catch (e) {
        if (e && e.name === 'AbortError') throw e;
      }
      return out;
    },
  };
}
