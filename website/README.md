# agent-sdk website

A single, self-contained landing page (`index.html`) — no build step. Tailwind
is loaded via CDN with an inline theme config (the `rust`/`ink` palettes).

## Deploy
- **GitHub Pages:** publish this directory (or copy `index.html` to the Pages root).
- **Vercel / Netlify:** point at this directory; static, no build command.
- **Local preview:** `python3 -m http.server` here, then open http://localhost:8000.

Status: **0.9 technical preview**. Bump the version badges in `index.html` when cutting a release.
