# GPT Pass

This folder contains the GPT-generated review and visualization pass for the Agent Server planning set.

Contents:

- `project-overview.md`
- `dependency-map.md`
- `storage-and-transport-addendum.md`
- `linear-project-audit-2026-04-08.md`
- `visuals/architecture-atlas-philosophy.md`
- `visuals/architecture-atlas.html`
- `visuals/shared/atlas.css`
- `visuals/shared/atlas.js`
- `visuals/sequence/01-root-turn-flow.html`
- `visuals/sequence/02-tool-runtime-flow.html`
- `visuals/sequence/03-checkpoint-recovery-flow.html`
- `visuals/sequence/04-event-replay-flow.html`
- `visuals/sequence/05-subagent-thread-flow.html`
- `visuals/state/01-root-task-state-machine.html`
- `visuals/state/02-tool-task-state-machine.html`
- `visuals/state/03-authority-and-commit-map.html`
- `visuals/state/04-schema-relationship-map.html`
- `visuals/output/architecture-atlas-preview.png`

The visualization set is now split intentionally:

- `visuals/architecture-atlas.html` is the hub.
- `visuals/sequence/` explains how work moves through time.
- `visuals/state/` explains lifecycle, ownership, and schema relationships.

These files are exploratory support artifacts for review and discussion. They should not be treated as the authoritative replacement plan on their own.

The current recommended reading order is:

1. `project-overview.md`
2. `dependency-map.md`
3. `storage-and-transport-addendum.md`
4. the architecture atlas and flow visuals
