# og image — /blog/does-xerj-beat-jev

Regenerate `landing/og/does-xerj-beat-jev.png` from
`og-blog-does-xerj-beat-jev.html` (this directory; NOT under landing/, which
deploys verbatim):

```sh
NODE_PATH=/root/pw/node_modules node - <<'EOF'
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
  await page.goto('file://' + process.cwd() + '/scripts/seo/assets/og-blog-does-xerj-beat-jev.html');
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(300);
  await page.screenshot({ path: 'landing/og/does-xerj-beat-jev.png' });
  await browser.close();
})();
EOF
```

Notes (the short list):
- Viewport must be exactly 1200x630, deviceScaleFactor 1 — same shape as
  `landing/og/xerj-card.png`. `mk_og_card.py --check` does NOT gate this file
  (it gates only xerj-card.png); the committed PNG is the render, so commit the
  exact playwright output and this source together.
- The palette is FIXED night (rasters have no theme); hexes read from
  style.css `:root`. If the brand tokens move, move them here too.
- Offline sandboxes: the Google Fonts link fails silently and the fallback
  stacks ('Arial Narrow'/Impact for display, Courier for mono) render instead.
  Re-render where the webfonts load before shipping, if the fallback was used.
- seo_lint rule 31 checks og:image is absolute https AND exists on disk under
  landing/ — pagedata's `og_image` key points at og/does-xerj-beat-jev.png,
  so the PNG must be committed for the lint to pass.
