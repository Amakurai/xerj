// Plumbing check: the console serves, the index exists, Discover finds a term
// that only the fixture contains. Product specs (corpus home, reader, graph
// hop, share claim) live beside this file and depend on the feature branches.
import { test, expect } from '@playwright/test';

test('engine holds the fixture and the console serves', async ({ page, request }) => {
  const cat = await request.get('/_cat/indices?format=json');
  expect(cat.ok()).toBeTruthy();
  const names = (await cat.json()).map((r) => r.index);
  expect(names.some((n) => n.startsWith('e2e-'))).toBeTruthy();

  const hits = await request.post('/e2e-*/_search', { data: { query: { match: { body: 'earnout' } }, size: 5 } });
  expect(hits.ok()).toBeTruthy();
  expect((await hits.json()).hits.total.value).toBeGreaterThan(0);

  await page.goto('/_xerj-console/');
  await expect(page).toHaveTitle(/xerj/i);
});
