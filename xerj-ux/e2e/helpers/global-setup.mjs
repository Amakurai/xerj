import { bootOpenNode, autoindex } from './node.mjs';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

export default async function globalSetup() {
  const fixture = process.env.XERJ_E2E_FIXTURE;
  if (!fixture) throw new Error('XERJ_E2E_FIXTURE must point at the folder to index (scripts/e2e.sh sets it)');
  const node = await bootOpenNode();
  const out = autoindex(node.url, fixture, ['--prefix', 'e2e', '--brain', 'e2e']);
  const state = { url: node.url, dataDir: node.dataDir, autoindexOut: out.slice(-2000) };
  writeFileSync(join(tmpdir(), 'xerj-e2e-state.json'), JSON.stringify(state));
  process.env.XERJ_URL = node.url;
  globalThis.__xerjNode = node; // same process as teardown under playwright's runner
}
