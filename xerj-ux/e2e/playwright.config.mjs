// End-to-end tests for the bundled console against a REAL xerj node.
//
// Run:  scripts/e2e.sh            (builds xerj-server scoped, boots a node on a
//                                   private port with a throwaway data dir,
//                                   indexes the fixture, runs the specs)
// The node lifecycle lives in helpers/node.mjs (globalSetup/teardown); every
// spec talks to process.env.XERJ_URL. Never point this at :9200.
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './specs',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['list']] : [['list']],
  globalSetup: './helpers/global-setup.mjs',
  globalTeardown: './helpers/global-teardown.mjs',
  use: {
    baseURL: process.env.XERJ_URL || 'http://127.0.0.1:9377',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
});
