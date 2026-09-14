// Boot / stop a throwaway xerj node for the e2e suite.
//
// Owner-path specs run the node in OPEN mode (`--insecure`, loopback bind) so
// the browser needs no login; the share-link specs boot a second, authed node
// through `xerj brain` because a public share must never exist on an open
// node. Both use a private port and a fresh data dir under the OS temp dir.
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export function xerjBinary() {
  const bin = process.env.XERJ_BIN;
  if (bin && existsSync(bin)) return bin;
  const guess = join(process.cwd(), '..', '..', 'engine', 'target', 'release', process.platform === 'win32' ? 'xerj.exe' : 'xerj');
  if (existsSync(guess)) return guess;
  throw new Error(`xerj binary not found; set XERJ_BIN or build: cd engine && cargo build --release -p xerj-server (looked at ${guess})`);
}

export async function waitReady(url, ms = 120_000) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${url}/health/ready`);
      if (r.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`node at ${url} not ready after ${ms}ms`);
}

/** Boot an open-mode node. Returns { url, port, dataDir, stop }. */
export async function bootOpenNode({ port = Number(process.env.XERJ_PORT || 9377) } = {}) {
  const dataDir = mkdtempSync(join(tmpdir(), 'xerj-e2e-'));
  const child = spawn(xerjBinary(), ['--insecure', '--data-dir', dataDir, '--port', String(port), '--bind', '127.0.0.1'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, XERJ_DISABLE_FEEDBACK: 'true' },
  });
  let log = '';
  child.stdout.on('data', (d) => { log += d; });
  child.stderr.on('data', (d) => { log += d; });
  const url = `http://127.0.0.1:${port}`;
  try {
    await waitReady(url);
  } catch (e) {
    child.kill('SIGKILL');
    throw new Error(`${e.message}\n--- server log ---\n${log.slice(-4000)}`);
  }
  return {
    url, port, dataDir,
    log: () => log,
    stop: () => { try { child.kill('SIGTERM'); } catch {} rmSync(dataDir, { recursive: true, force: true }); },
  };
}

/** Index a folder into the node with the real autoindex pipeline (graph on). */
export function autoindex(url, folder, extra = []) {
  const out = execFileSync(xerjBinary(), ['autoindex', folder, '--url', url, '--progress', 'none', ...extra], {
    encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, XERJ_DISABLE_FEEDBACK: 'true' },
  });
  return out;
}
