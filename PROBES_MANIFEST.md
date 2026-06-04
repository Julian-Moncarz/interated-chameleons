# Probe Manifest

All trained probes pulled from the GPU boxes (`cambria-zakim`, `cambria-union`),
organized by the model/run that produced them. Probe files are `<concept>_probe_<N>.pt`
where `<N>` is the seed/round index. Each `probes/` dir has its own
`probe_metadata.json` with per-concept train AUROC.

**Total: 93 probe files (~1.45 MB) across 5 sets.** (Weights are gitignored — see README.)

| Set | Path | Files | Concepts | Rounds/seeds | Source box | Notes |
|---|---|---|---|---|---|---|
| Scraped concept probes | `checkpoints/scraped/probes/` | 11 | 11 | `_0` | zakim & union (identical) | Frozen concept probes for the **Scraped** chameleon (default model). |
| Generated concept probes | `checkpoints/generated/probes/` | 11 | 11 | `_0` | zakim | Frozen probes for the **Generated** chameleon. Uses `chemistry-based`, `literature-focused` (vs `chemistry`, `lesswrong` in scraped). |
| Arms-race adversary probes | `checkpoints/arms_race/probes/` | 66 | 11 | `_0`..`_5` | union | **Q2 iterated adversary probes** — `_N` = arms-race round 0–5, 11 concepts × 6 rounds. The unique Q2 artifact. |
| MVP probes *(archived)* | `archive/checkpoints_mvp/probes/` | 3 | 3 | `_0` | zakim | Retired 3-concept generated run: german, literature-focused, mathematical. |
| Smoke probes *(archived)* | `archive/checkpoints_smoke/probes/` | 2 | 2 | `_0` | zakim | Retired 2-concept sanity test: allcaps, biology-focused. |

## Concept sets

- **Scraped / Arms-race (11):** allcaps, biology-focused, chemistry, comforting, confused,
  finnish, german, html, jokey, lesswrong, mathematical
- **Generated (11):** allcaps, biology-focused, **chemistry-based**, comforting, confused,
  finnish, german, html, jokey, **literature-focused**, mathematical
- **MVP (3):** german, literature-focused, mathematical
- **Smoke (2):** allcaps, biology-focused

> Naming difference between data sources: **scraped/arms-race** use `chemistry` and `lesswrong`;
> **generated/mvp** use `chemistry-based` and `literature-focused`. Filenames reflect this —
> don't assume they're interchangeable.

## Train AUROC (from each set's metadata)

- Scraped: 0.9936–1.0000 (mean 0.9989)
- Generated: 1.0000 (all concepts)
- MVP: 1.0000
- Smoke: 1.0000
- Arms-race: adversary re-trains to ~1.0 every round (see `checkpoints/arms_race/arms_race_log.json`)
