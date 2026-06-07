# agent-sdk — Brand Guide

The visual identity for **agent-sdk**: a Rust toolkit for building production AI
agents. The brand is geometric, calm, and engineering-grade — a clean mark,
one confident accent color, and disciplined type.

---

## 1. Logo — the "hexloop"

The mark is a **rounded-corner hexagon housing a directional agent loop**.

- The **hexagon** is the self-contained unit — a single, well-bounded agent (and
  a nod to the modular, multi-crate workspace it ships as).
- The **circular arrow inside** is the agent loop: _act → observe → repeat_. The
  single crisp arrowhead shows it turning — autonomous, iterative, always
  running. One unit, one loop.

### Files

| File | Use | Color |
| --- | --- | --- |
| `agent-sdk-mark.svg` | **Master.** `currentColor` — recolors via CSS/context. | inherits `color` |
| `agent-sdk-mark-indigo.svg` | Primary mark on light surfaces. | `#4F46E5` |
| `agent-sdk-mark-dark.svg` | Mark on dark surfaces. | `#F5F5F7` |
| `agent-sdk-favicon.svg` | Tab/app icon, optimized to read at 16px. | `#4F46E5` |
| `agent-sdk-lockup.svg` | Horizontal mark + wordmark (`currentColor`). | inherits `color` |
| `agent-sdk-lockup-indigo.svg` | Lockup: indigo mark + near-black wordmark. | `#4F46E5` / `#0D0D0F` |
| `agent-sdk-readme-header.svg` | 1280×320 README / repo banner. | dark, indigo glow |

> The master and lockup use `fill="none" stroke="currentColor"` (and
> `fill="currentColor"` for the arrowhead). Set the CSS `color` property — or an
> SVG `color` — to recolor without editing the file. All marks are transparent;
> no baked background.

### The favicon is intentionally different

The directional arrowhead is the most expressive part of the mark, but it
collapses into noise below ~24px. The **favicon therefore drops the arrowhead
and expresses the loop as a clean ring with a single node dot sitting in the
top gap** — "a node on its loop." It also uses a slightly heavier stroke (2.6 vs
2.2) so the three forms (hexagon, ring, dot) stay distinct at 16px. Same
concept, tuned for tiny. Use the full directional mark everywhere ≥ 24px.

### Clearspace

Keep clear space around the mark equal to **half the mark's height** (½×) on all
sides. In the lockup, the gap between mark and wordmark is ~½ the mark height;
do not tighten it.

### Minimum sizes

| Asset | Min size |
| --- | --- |
| Full directional mark | **24px** (below this, use the favicon) |
| Favicon (ring + node) | **16px** |
| Lockup | **96px** wide |

### Do

- Recolor the master via `currentColor` (indigo on light, near-white on dark).
- Keep the hexagon, ring, and arrowhead at one consistent stroke weight.
- Give it room — respect clearspace.

### Don't

- Don't recolor the loop and hexagon differently from each other.
- Don't add a drop shadow, bevel, gradient, or outline to the mark.
- Don't stretch, rotate, or reflect it (the loop direction is meaningful).
- Don't place the indigo mark on a low-contrast surface — switch to the dark/
  near-white variant instead.
- Don't use the arrowhead version below 24px — use the favicon.

---

## 2. Color

### Primary — Indigo

The single brand accent. Use it sparingly: the mark, key links, primary actions.

| Token | Hex | Use |
| --- | --- | --- |
| **Indigo 600 — Primary** | `#4F46E5` | The brand color. Mark, primary buttons, active states. |

#### Indigo ramp (tints → shades)

| Token | Hex |
| --- | --- |
| Indigo 50 | `#EEEDFC` |
| Indigo 100 | `#DAD8F9` |
| Indigo 200 | `#B8B4F2` |
| Indigo 300 | `#938DEC` |
| Indigo 400 | `#7C75F0` |
| **Indigo 500** | `#6259E9` |
| **Indigo 600 (Primary)** | `#4F46E5` |
| Indigo 700 | `#3F37C9` |
| Indigo 800 | `#322B9E` |
| Indigo 900 | `#272178` |

> `Indigo 400 (#7C75F0)` is the on-dark accent (used for the mark in the README
> banner) — indigo 600 is a touch dense against deep ink.

### Ink & Paper

| Token | Hex | Use |
| --- | --- | --- |
| **Ink** | `#0D0D0F` | Primary text on light; near-black wordmark. |
| **Ink (banner bg)** | `#0D0D12` → `#15131F` | Dark-surface background (subtle indigo-leaning gradient). |
| **Paper** | `#FFFFFF` | Primary light background. |
| **Paper alt** | `#FAFAFA` | Secondary / inset light background. |
| **Ink invert** | `#F5F5F7` | Near-white text / mark on dark. |

### Neutral gray scale

| Token | Hex |
| --- | --- |
| Gray 50 | `#FAFAFA` |
| Gray 100 | `#F4F4F5` |
| Gray 200 | `#E4E4E7` |
| Gray 300 | `#D4D4D8` |
| Gray 400 | `#A1A1AA` |
| Gray 500 | `#71717A` |
| Gray 600 | `#52525B` |
| Gray 700 | `#3F3F46` |
| Gray 800 | `#27272A` |
| Gray 900 | `#18181B` |

> `#A9A7B8` is the muted lavender-gray used for the tagline on the dark banner —
> a hint of indigo keeps secondary text on-brand against deep ink.

### Light / dark usage

- **Light:** Paper `#FFFFFF` (or Paper alt `#FAFAFA`) background, Ink `#0D0D0F`
  text, Indigo 600 `#4F46E5` accent + mark.
- **Dark:** Ink `#0D0D12` background, Ink-invert `#F5F5F7` text, Indigo 400
  `#7C75F0` accent + mark.
- Keep indigo for emphasis only — lean on the gray scale for structure so the
  accent stays meaningful.

---

## 3. Typography

### Geometric sans — product & wordmark

**Inter** (preferred) or **Geist**. Used for the wordmark, UI, headings, and
body. The wordmark is **lowercase `agent-sdk`, weight 600, slightly tight
tracking** — keep the hyphen.

```
font-family: "Inter", ui-sans-serif, system-ui, -apple-system, sans-serif;
```

| Role | Size / line-height | Weight | Tracking |
| --- | --- | --- | --- |
| Wordmark | optical | 600 | −0.04em |
| Display / H1 | 40 / 48 | 600 | −0.02em |
| H2 | 28 / 36 | 600 | −0.015em |
| H3 | 20 / 28 | 600 | −0.01em |
| Body | 16 / 26 | 400 | 0 |
| Small / caption | 14 / 20 | 400 | 0 |
| Eyebrow / label | 12 / 16 | 500 | +0.06em, UPPERCASE |

### Monospace — code

**JetBrains Mono** (preferred) or **Geist Mono**. Code blocks, inline `code`,
terminal output, version strings.

```
font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
```

| Role | Size / line-height | Weight |
| --- | --- | --- |
| Code block | 14 / 22 | 400 |
| Inline code | 0.9em of surrounding | 400–500 |

> Pairing rule: Inter/Geist for everything human-readable, JetBrains/Geist Mono
> for everything machine-shaped. Both are open-source and widely available
> (Google Fonts / npm / system fallbacks), keeping the SVGs renderable without
> bundling fonts.
