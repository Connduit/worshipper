# Worshipper dashboard

React + TypeScript + Vite control-plane prototype.

```sh
cd dashboard
npm ci
npm run dev -- --host 0.0.0.0
npm run build
npm run lint
npm test
```

The interface currently uses explicitly labeled demo results and browser-local run drafts/preferences. It does not execute agents, contact inference endpoints, or change server settings. Do not enter credentials: browser storage is not secret storage.

The next integration needs a local backend with authenticated run creation, structured run events, cancellation, model health, and project-scoped memory. Keep shell execution and policy enforcement in that backend; do not expose arbitrary shell commands through browser controls.

Browser tests use `/usr/bin/chromium` by default; set `CHROMIUM_PATH` to your installed Chromium executable on other systems.
