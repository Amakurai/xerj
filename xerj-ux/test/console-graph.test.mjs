// The console's graph read path (issue #936): a signed-in operator reads
// brains through the console's own session-authorized endpoints, never
// through the data plane (whose `/_graph/*` and reserved-namespace
// `/_search` routes 401 a session on an auth-enabled engine — the default).
//
// Run: node --test xerj-ux/test/
import test from 'node:test';
import assert from 'node:assert/strict';

// second-brain-api.js reads location.hash while assembling a moment; the
// rest of its browser surface is guarded (document, storage, rAF).
globalThis.location = { hash: '' };

const calls = [];
let respond = () => json(404, {});
globalThis.fetch = async (url, init) => { calls.push({ url: String(url), init }); return respond(String(url), init); };
const json = (status, body) => ({ status, ok: status >= 200 && status < 300, json: async () => body, text: async () => (typeof body === 'string' ? body : JSON.stringify(body)) });

const { makeConsoleTransport, BRAINS_WALKED } = await import('../src/data/transport-console.js');
const { sbBrainsPresent, invalidateBrainsProbe } = await import('../src/data/brains-probe.js');
const { liveSecondBrain } = await import('../src/data/second-brain-api.js');

test('the reader transport reads the graph through the console session, same-origin', async () => {
  calls.length = 0;
  respond = (url) => {
    assert.match(url, /^\/_xerj-console\/api\/v1\/graph\/casefile\/ego\?node=file-01&hops=1&direction=both/, url);
    return json(200, { brain: 'casefile', edges: [] });
  };
  const r = await makeConsoleTransport().ego('casefile', { node: 'file-01', hops: '1', direction: 'both' }, undefined);
  assert.equal(r.status, 200);
  assert.equal(r.body.brain, 'casefile');
  assert.equal(calls[0].init.credentials, 'same-origin', 'the session cookie must ride the console read');
});

test('brain discovery is ONE listing call that says which nodes index each brain feeds', async () => {
  calls.length = 0;
  respond = () => json(200, { data: { brains: [
    { name: 'casefile', nodes_index: 'ax-docs' },
    { name: 'vendors', nodes_index: 'ax-mail,ax-pdfs' },
    { name: 'other', nodes_index: 'ax-mail' },
    { name: 'no-dataset', nodes_index: '.xerj-memory-no-dataset' },
  ], total: 4 } });
  const t = makeConsoleTransport();
  assert.deepEqual(await t.discoverBrains('ax-docs', undefined), ['casefile']);
  assert.deepEqual(await t.discoverBrains('ax-pdfs', undefined), ['vendors']);
  assert.deepEqual(await t.discoverBrains('ax-mail', undefined), ['vendors', 'other']);
  assert.deepEqual(await t.discoverBrains('nothing-lists-this', undefined), []);
  assert.equal(calls.length, 4, 'one listing per question — no _cat walk, no meta-doc reads');
  for (const c of calls) {
    assert.equal(c.url, '/_xerj-console/api/v1/graph/brains');
    assert.equal(c.init.credentials, 'same-origin');
  }
  // A role that may not read brains gets [] — the reader then says "no
  // brain for this index", which is the honest statement for that role.
  calls.length = 0;
  respond = () => json(403, { error: { reason: 'the console role "viewer" may not read brains' } });
  assert.deepEqual(await t.discoverBrains('ax-docs', undefined), []);
});

test('the listing caps the walk at BRAINS_WALKED brains, in listing order', async () => {
  const many = Array.from({ length: BRAINS_WALKED + 3 }, (_, i) => ({ name: `b-${i}`, nodes_index: 'ax-docs' }));
  respond = () => json(200, { data: { brains: many, total: many.length } });
  const out = await makeConsoleTransport().discoverBrains('ax-docs', undefined);
  assert.equal(out.length, BRAINS_WALKED);
  assert.deepEqual(out, many.slice(0, BRAINS_WALKED).map((b) => b.name));
});

test('the nav probe claims a brain only when the listing proves one', async () => {
  invalidateBrainsProbe();
  respond = () => json(200, { data: { brains: [{ name: 'casefile', nodes_index: 'ax-docs' }], total: 1 } });
  assert.equal(await sbBrainsPresent('http://eng-a:9200'), true);
  respond = () => json(403, {});
  assert.equal(await sbBrainsPresent('http://eng-b:9200'), false, '403 is not "no brains on this node", but it is no proof of one');
  respond = () => json(200, { data: { brains: [], total: 0 } });
  assert.equal(await sbBrainsPresent('http://eng-c:9200'), false);
  respond = () => { throw new TypeError('Failed to fetch'); };
  assert.equal(await sbBrainsPresent('http://eng-d:9200'), false, 'engine unreachable → no claim');
  assert.equal(await sbBrainsPresent(''), false);
});

// ── the Second Brain dashboard's own reads ─────────────────────────────────
//
// A full liveSecondBrain() pass over a multi-dataset brain ("ax-mail,ax-pdfs"):
// every fetch must go through the console graph endpoints — except the
// searches-per-index tile's /v1/metrics, which stays a data-plane read by
// decision and shows its refusal honestly when it is refused.

const OVERVIEW = {
  brain: 'casefile', contract: 'xerj-second-brain/1', exists: true,
  nodes_index: 'ax-mail,ax-pdfs', nodes: { total: 14 }, embedder: 'lexical-feature-hash',
  edges: { total: 3, live: 2, invalidated: 1 },
  types: [], detectors: [],
  hubs: { out: [{ id: 'file-01', live_edges: 2 }], in: [] },
  created_over_time: [], not_shown: {},
};

const egoBody = (qs) => ({
  brain: 'casefile', node: qs.get('node'), edges: [
    { edge_id: 'e1', src: 'file-01', dst: 'file-02', type: 'same_dir', valid_at: 1753600000000, invalid_at: null },
    { edge_id: 'e3', src: 'file-02', dst: 'file-03', type: 'pathcite', valid_at: 1753600000000, invalid_at: 1753650000000 },
  ],
  nodes: { 'file-01': { title: '01.eml' } }, neighbors: [], not_shown: {},
});

function graphRouter(url) {
  if (url === '/_xerj-console/api/v1/graph/brains') {
    return json(200, { data: { brains: [{ name: 'casefile', nodes_index: 'ax-mail,ax-pdfs', contract: 'xerj-second-brain/1' }], total: 1, not_shown: { brains_clipped: 0 } } });
  }
  if (url.startsWith('/_xerj-console/api/v1/graph/casefile/overview')) return json(200, OVERVIEW);
  if (url.startsWith('/_xerj-console/api/v1/graph/casefile/ego?')) {
    return json(200, egoBody(new URLSearchParams(url.slice(url.indexOf('?') + 1))));
  }
  return json(404, {});
}

test('the dashboard reads everything through the console graph endpoints', async () => {
  calls.length = 0;
  respond = (url, init) => {
    const body = init && init.body ? JSON.parse(init.body) : null;
    if (url.startsWith('/_xerj-console/api/v1/graph/casefile/edges/_search')) {
      if (body && body.query && body.query.exists && body.query.exists.field === 'invalid_at') {
        return json(200, { hits: { total: { value: 1, relation: 'eq' }, hits: [{ _index: '.xerj-memory-casefile-edges', _id: 'e3', _source: { src: 'file-02', dst: 'file-03', type: 'pathcite' } }] } });
      }
      if (body && body.query && body.query.exists && body.query.exists.field === 'src_format') {
        return json(200, { hits: { total: { value: 3, relation: 'eq' } }, aggregations: { src: { buckets: [{ key: 'eml', doc_count: 2, dst: { buckets: [{ key: 'eml', doc_count: 2 }] } }] } } });
      }
      return json(400, {});
    }
    if (url.startsWith('/_xerj-console/api/v1/graph/casefile/nodes/_search')) {
      const which = new URLSearchParams(url.slice(url.indexOf('?') + 1)).get('index');
      if (body && body.aggs && body.aggs.formats) {
        return json(200, which === 'ax-mail'
          ? { hits: { total: { value: 10, relation: 'eq' } }, aggregations: { formats: { buckets: [{ key: 'eml', doc_count: 6 }, { key: 'pdf', doc_count: 2 }] } } }
          : { hits: { total: { value: 4, relation: 'eq' } }, aggregations: { formats: { buckets: [{ key: 'pdf', doc_count: 3 }] } } });
      }
      // name hydration (ids query)
      return json(200, { hits: { total: { value: 3, relation: 'eq' }, hits: [
        { _id: 'file-01', _source: { title: '01.eml', ax_path: 'inbox/01.eml' } },
        { _id: 'file-02', _source: { title: '02.eml', ax_path: 'inbox/02.eml' } },
        { _id: 'file-03', _source: { title: '03.eml', ax_path: 'inbox/03.eml' } },
      ] } });
    }
    if (url === 'http://eng:9200/v1/metrics') {
      return json(200, 'xerj_queries_by_index_total{index="ax-mail"} 7\nxerj_queries_by_index_total{index="ax-pdfs"} 2\n');
    }
    return graphRouter(url);
  };

  const data = await liveSecondBrain('http://eng:9200', {}, undefined);

  // WHAT was read: only console graph endpoints + the one metrics read.
  const urls = calls.map((c) => c.url);
  for (const u of urls) {
    assert.ok(
      u.startsWith('/_xerj-console/api/v1/graph/') || u === 'http://eng:9200/v1/metrics',
      `the dashboard must not read the data plane directly: ${u}`,
    );
    assert.ok(!u.includes('.xerj-memory-') && !u.includes('/_graph/') && !u.includes('/_cat'), u);
  }
  for (const c of calls) {
    if (c.url.startsWith('/_xerj-console')) {
      assert.equal(c.init.credentials, 'same-origin', `${c.url} must carry the session cookie`);
    }
  }
  assert.ok(urls.includes('/_xerj-console/api/v1/graph/brains'));
  assert.ok(urls.some((u) => u.includes('/casefile/overview?')));
  assert.equal(urls.filter((u) => u.includes('/casefile/ego?')).length, 2, 'belief (2-hop) + belief strip (1-hop)');
  assert.equal(urls.filter((u) => u.includes('/edges/_search')).length, 2, 'recent retirements + file-type crossings');

  // The multi-dataset fan-out names WHICH of the brain's indices it reads.
  const nodeSearchUrls = urls.filter((u) => u.includes('/nodes/_search'));
  assert.equal(nodeSearchUrls.length, 4, 'notes tally + name hydration, one per dataset index');
  assert.deepEqual(nodeSearchUrls.map((u) => new URLSearchParams(u.slice(u.indexOf('?') + 1)).get('index')).sort(), ['ax-mail', 'ax-mail', 'ax-pdfs', 'ax-pdfs'],
    'a multi-name nodes_index passes ?index= — the endpoint is pinned to ONE of the brain\'s own indices');

  // And the answers are merged client-side, ran numbers only.
  assert.deepEqual(data.sb.brains, ['casefile']);
  assert.equal(data.overview.edges.total, 3);
  assert.equal(data.sb.nodeStats.total, 14, '10 on ax-mail + 4 on ax-pdfs');
  assert.deepEqual(data.sb.nodeStats.formats, [
    { format: 'eml', count: 6 }, { format: 'pdf', count: 5 },
  ], 'per-format buckets merged across the two indices, count desc');
  assert.equal(data.sb.crossings.total, 3);
  assert.deepEqual(data.sb.serverReads.rows, [
    { index: 'ax-mail', count: 7 }, { index: 'ax-pdfs', count: 2 },
  ]);
  assert.equal(data.sb.recentRetired.length, 1);
  assert.equal(data.sb.recentRetired[0].edge_id, 'e3');
  assert.equal(data.sb.names['file-02'].title, '02.eml', 'hub ids hydrated to titles');
  assert.equal(data.sb.focus, 'file-01', 'default focus = strongest hub');
});

test('a role that may not read brains gets an honest empty state, not a leak or a loop', async () => {
  calls.length = 0;
  respond = () => json(403, { error: { type: 'graph_error', reason: 'the console role "viewer" may not read brains' } });
  const data = await liveSecondBrain('http://eng:9200', {}, undefined);
  assert.deepEqual(data.sb.brains, []);
  assert.equal(data.brain, null);
  assert.equal(data.overview, null, 'no brain, no overview read — nothing further was fetched');
  assert.equal(calls.filter((c) => c.url !== '/_xerj-console/api/v1/graph/brains').length, 0);
});
