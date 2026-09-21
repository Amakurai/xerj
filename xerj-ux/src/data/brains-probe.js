// ============================================================
// Xerj Console — "does at least one brain exist?" probe
//
// The Second Brain dashboard only earns a nav entry once the engine
// actually holds a brain — a console pointed at an engine nobody ever ran
// `xerj brain <folder>` against should not advertise an empty dashboard.
// This module answers exactly that yes/no question.
//
// It asks the console's own brains listing
// (`/_xerj-console/api/v1/graph/brains`), authenticated by the console
// session. The old direct `_cat/indices/.xerj-memory-*` call 401s a session
// on an auth-enabled engine (the default), which kept the nav entry hidden
// for a signed-in operator exactly when the dashboard finally had a
// session-authorized way to read the graph (issue #936).
//
// Honesty stance: `false` on ANY transport failure. An unreachable engine
// — or a role that may not read brains (403) — means we cannot claim a
// brain exists, so the nav entry stays hidden; the deep-link route still
// resolves (app.js never filters the route table, only the nav list), so
// nothing is lost.
//
// Results are cached per baseUrl with a short TTL, so:
//   - the boot probe + periodic re-probes don't hammer the listing,
//   - switching backend/base-URL naturally misses the cache and
//     re-probes the new target without an explicit invalidation hook.
// ============================================================

const TTL_MS = 30_000;

/** baseUrl → { at: epoch-ms, value: boolean } */
const cache = new Map();

/** Drop all cached answers (e.g. right after `xerj brain` finishes,
 *  or when settings change under us). Safe to call any time. */
export function invalidateBrainsProbe() {
  cache.clear();
}

/**
 * True iff this engine holds at least one brain (a reserved
 * `.xerj-memory-{brain}-edges` index) that the console session may read.
 *
 * Never throws; returns false on transport failure, non-OK status, or an
 * empty/absent baseUrl. The graph listing is same-origin (the console
 * serves the SPA), so `baseUrl` only keys the cache.
 */
export async function sbBrainsPresent(baseUrl, signal) {
  const base = (baseUrl || '').replace(/\/+$/, '');
  if (!base) return false;

  const hit = cache.get(base);
  if (hit && Date.now() - hit.at < TTL_MS) return hit.value;

  let present = false;
  try {
    const r = await fetch('/_xerj-console/api/v1/graph/brains', {
      signal,
      credentials: 'same-origin',
      headers: { accept: 'application/json' },
    });
    // 403 = this console role may not read brains; 401 = no session. Both
    // are "cannot prove a brain exists", so we do not claim one.
    if (r.ok) {
      const j = await r.json();
      present = ((j && j.data && j.data.brains) || []).length > 0;
    }
  } catch {
    present = false; // engine down / CORS / abort — no claim
  }

  cache.set(base, { at: Date.now(), value: present });
  return present;
}
