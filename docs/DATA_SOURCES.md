# Data sources and provenance

Author: Md Forhadul Islam

Updated: 29 October 2026

## Selected study data

The completed study uses the [Betty Dota 2 Decision Context Dataset](https://huggingface.co/datasets/wolframko/betty-dota2), published by wolframko. The publisher credits Valve replay files, STRATZ metadata and enrichment, and the manta replay parser. These data and their collection are not a contribution of this independent study.

The study transfers parts of the workflow of Dr Fazilat Hojaji and her coauthors to another game. Betty is not their League of Legends dataset, and results obtained from it cannot establish numerical reproduction of their findings.

| Provenance field | Recorded value |
|---|---|
| Publisher | wolframko |
| Download date | 29 October 2026 |
| Pinned dataset revision | `b20e01577af2f4f5dffcdc4f1d5bad5c3807abca` |
| Raw destination | `data/raw/betty-dota2/` |
| Download implementation | `src/download_dota2.py` |
| Inventory and hash verification | `outputs/results/D_download_manifest.json` |
| Download status | Complete, with no recorded failures |
| Publisher declared licence | MIT |
| Redistribution approach | Raw data remain excluded from Git |

The manifest records the exact file inventory, expected and verified bytes, hashes, completion time and verification method. It reports matching expected and verified bytes. Publisher hashes were checked where supplied; local hashes provide provenance for other files. The downloader can verify existing files before resuming.

## Licence and upstream ownership

The publisher card declares MIT. The pinned dataset snapshot does not provide an additional standalone licence text. Preserve the card and attribution, and describe the licence as the publisher's declaration. It does not establish ownership of Valve or STRATZ material or grant rights to redistribute game assets. This project publishes analysis code and results while keeping the downloaded raw files out of the repository.

The card's linked source code branch was unavailable during the literature provenance check. The dataset can be inspected and analysed, but the publisher's complete parsing pipeline was not independently verified. The review ledger records this limitation in `outputs/results/D_literature_sources.json`.

## What the audit established

The metadata audit found 9,385 unique match identifiers without duplicate identifiers. Recorded match dates span February 2024 to December 2025. The actual date range is broader than the card's description of professional leagues in 2025. These observations come from `outputs/results/D_data_audit.json`, produced by `src/audit_dota2.py`.

All file schemas and table row totals were inventoried. Detailed field missingness and sampling checks used the first shard of each inspected table family, rather than every row of every family. Sampled hero states contain coordinates, experience and net worth. Sampled ward coordinates are absent, some inventory values are generic `CDOTA_Item` strings, and some role fields are empty. These findings limit detailed vision, item and role interpretation. A field advertised by the dataset card is not automatically a complete usable variable.

Later preparation and integrity checks are recorded separately:

| Evidence file | Purpose |
|---|---|
| `D_preparation.json` | Observed features, eligible snapshots and partition counts |
| `D_clock_alignment.json` | Replay and combat clock alignment checks |
| `D_identity_audit.json` | Hero identity and team assignment checks |
| `D_integrity_validation.json` | Draft consistency, outcome consistency, temporal rules and partition integrity |
| `D_protocol.json` | Frozen analysis choices and completed final evaluation status |

These files are under `outputs/results/`. Internal consistency checks compare dataset fields and are not independent ground truth. Missing draft arrays are explicitly recorded rather than counted as successful checks.

## Timing, identity and allowed inputs

Replay `game_time` and combat log timestamps can differ because of pauses. Preparation uses preceding combat timestamp anchors to align raw tick states with the match clock. It selects information available by the observation time and excludes completed matches at that time. Remaining timing resolution limits are documented in the protocol and integrity outputs.

Replay slot order does not reliably identify Radiant and Dire. Observed hero names are mapped through the pinned OpenDota dictionary to metadata draft identifiers, with distinct hero and team membership checks. Blank hero names are excluded. Identifiers used for alignment are not prediction features.

Final outcome, eventual match duration, final statistics and retrospective role labels are excluded from prospective model inputs. Team feature differences are constructed from observed state. Later observation times describe the matches still active and sufficiently observed then.

## OpenDota hero dictionary

The identity mapping credits **The OpenDota Project and contributors**. Source: [pinned hero dictionary](https://github.com/odota/dotaconstants/blob/bf193a550f778dec35debf4d71d05a86bebcc418/build/heroes.json). The pinned [MIT licence](https://github.com/odota/dotaconstants/blob/bf193a550f778dec35debf4d71d05a86bebcc418/LICENSE) was inspected separately.

Download date: 29 October 2026. Revision: `bf193a550f778dec35debf4d71d05a86bebcc418`. Exact source URL, hash and licence provenance are in `outputs/results/D_hero_constants_source.json`. This dictionary is an implementation resource, not a match dataset or a research paper.

## Earlier League of Legends feasibility work

The [Zenodo dataset](https://zenodo.org/records/8303397) by Jailson Barros da Silva Junior and Claudio Campelo was considered before Dota 2 was selected. Its publisher API declared CC BY 4.0 on 29 October 2026. It is not Dr Hojaji's dataset and was not used for the completed Dota 2 model experiments.

An earlier inspection fetched `full_data_20.csv` into memory without retaining the raw file. The audit in `outputs/results/A_data_feasibility.json` records its schema, match identifiers and missingness. Gold, average player level, combat and objective statistics supported a reduced prediction task, but detailed movement, vision and several contextual fields were unavailable. The `timePercent` values could not safely be interpreted from the filename alone. Snapshots defined as percentages of eventual duration do not establish fixed minute prospective predictions.

Riot's API, Oracle's Elixir, a Kaggle League of Legends candidate and the ESTA Counter Strike dataset were also considered. They were not selected or used in the completed study. Their availability and any future use require a fresh source and licence check. Historical access failures do not establish their current availability.

Rugby statistics and video work were deferred. No rugby dataset or video footage was used.

## Completed analysis and deferred scope

Prediction, calibration, explanation stability and the movement sampling comparison have been executed. The protocol was frozen before the final evaluation, which is complete. Earlier invalid development diagnostics were withdrawn and are not study findings.

Hero context, detailed item builds and spatial ward coverage remain deferred because provenance and completeness did not support defensible prospective extensions. Professional match data do not establish generalisation across ordinary player ranks, regions, unseen teams or game patches.
