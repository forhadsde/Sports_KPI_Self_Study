# Study questions, evidence and completed sequence

Author: Md Forhadul Islam, London

Updated: 29 October 2026

## Scope and credit

This completed independent study adapts selected parts of the KPI workflow of Dr Fazilat Hojaji, Robert E. McIlroy, Antoine Dupuy, Gianpaolo Pedroni, Adam J. Toth and Mark J. Campbell to Dota 2. Their original method and findings remain fully credited. The study is not a numerical reproduction of their League of Legends results and does not claim affiliation with their group.

[Original paper](https://doi.org/10.1016/j.chbr.2025.100718). [Betty dataset](https://huggingface.co/datasets/wolframko/betty-dota2), published by wolframko using Valve replays and STRATZ data. OpenDota, scikit learn, PyTorch and the model and explanation method authors are credited in the literature record.

The core prediction, calibration, explanation stability and movement measurement analyses have been executed. Final evaluation followed the frozen protocol and integrity checks, without further tuning. Hero context, rugby and video work are deferred.

## Problem

A model can predict well while its feature importance order changes across models, samples or seeds. Predicted probabilities can also be unreliable despite reasonable classification accuracy. Movement estimates depend on observation frequency. This study examines these dependencies as predictive associations and measurement properties, without asserting causes of winning.

A KPI is a candidate performance measurement. Calibration means agreement between predicted probabilities and observed win frequencies. SHAP assigns feature contributions to a model prediction under specified assumptions. Bootstrap analysis repeatedly resamples matches to describe uncertainty.

## Reference adaptation

Expectations below record the questions considered before observing results. They are not substituted for the results.

| Question | Completed analysis | Original expectation and reason | Evidence |
|---|---|---|---|
| A1. Can an adapted network predict Dota 2 outcomes? | Reference network fitted at fixed observation times using prospective features | Team statistics might contain predictive information; establish a reference before reliability checks | `D_protocol.json`, `D_final_metrics.json` |
| A2. Which available statistics contribute to predictions? | SHAP explanations and global ranking summaries on development data | Economy and objectives were plausible indicators, but no order was assumed | `D_explanations.json`, `D_summary.json` |
| A3. Do rankings change as a match develops? | Explanations compared across prespecified observation times | Changing game state might change importance | `D_explanations.json`, `D_summary.json` |

All evidence files in this roadmap are under `outputs/results/`. Agreement with the original paper is descriptive because the game, data and features differ. There is no target accuracy that would prove numerical reproduction.

The fixed observation times are 600, 1200 and 1800 seconds, as recorded in the protocol. Each horizon includes only matches still active with eligible preceding observations. The common final match cohort analysis is reported separately because differing eligible populations can influence apparent changes over time.

## Core reliability checks

| Question | Completed analysis | Why it matters | Evidence |
|---|---|---|---|
| B1. Does model choice affect prediction or explanation? | Logistic regression, Random Forest, XGBoost and reference network compared on identical partitions | Separate evidence in the data from dependence on one model family | `D_final_metrics.json`, `D_summary.json` |
| B2. Are win probabilities reliable? | Raw and temperature adjusted probabilities reported with reliability bins, Brier score and log loss | Probability estimates should have an empirical interpretation | `D_calibration_bins.json`, `D_final_metrics.json` |
| B3. How stable are performance and KPI rankings? | Seed comparisons, fixed model explanation bootstrap, exploratory training refits and performance intervals | Quantify uncertainty rather than report one ranking as certain | `D_explanations.json`, `D_refit_stability.json`, `D_summary.json` |

Calibration and stability are supporting checks proposed for this study, rather than future directions explicitly requested by the original authors. Good discrimination does not guarantee calibration. Temperature scaling is an established method whose benefit was tested, not assumed.

Seed variation measures sensitivity to training randomness. A fixed model bootstrap changes the explained match sample without retraining. A training refit bootstrap changes the training sample and refits the model. These uncertainties must remain separate. Correlated features can share or exchange attribution, so rank comparisons do not establish causal importance.

The primary final comparison and its paired match and UTC day block uncertainty summaries were specified in the frozen protocol. UTC date blocks are a dependence sensitivity check, not verified tournament series. Test results were not used to choose the preferred probability adjustment.

The staged feature group ablation involving gold, experience, objectives, combat and vision was removed from the scope and was not restored.

## Completed movement measurement experiment

### C1. How does sampling frequency change measured movement?

The same eligible development trajectories were compared under native sampling and coarse sampling at multiple offsets. Continuous segment eligibility and endpoints were matched, with rules for death, missing intervals and large jumps. The experiment uses training and validation matches, rather than final test outcomes.

The expectation was that coarse samples could miss intervening movement. The useful evidence concerns the magnitude and variation of this measurement difference. Its direction alone is not a new discovery.

This question relates to sparse positional sampling identified in the original paper's limitations, but it does not estimate the corresponding effect in League of Legends. It also does not measure how movement sampling altered win prediction accuracy or KPI rankings in the fitted prediction models.

Evidence: `D_movement.json` and `D_movement_match_pairs.parquet`. Native sampling is not continuous ground truth. Speed based filtering may remove genuine fast movement or miss some ability movement. The reported quantity is sampled path displacement under operational rules, not true walking distance.

## Deferred context extension

### C2. Does hero or role context change KPI interpretation?

This extension was considered but deferred. Role labels may be inferred from later match behaviour, some role values are empty, and generic inventory strings prevent reliable detailed item interpretation. A prospective contextual experiment requires defensible labels available at the prediction time and sufficient subgroup coverage.

Hero identities used for team alignment do not establish validated behavioural role labels. The OpenDota dictionary supports identity joins, not an automatic causal or role classification claim. Spatial ward coverage was also deferred because sampled ward coordinates were absent.

Champion classes, items and composition are explicit future directions in the original paper. The completed study documents why these were not answered here instead of claiming to resolve them. The decision is frozen in `D_protocol.json`.

## Completed sequence and safeguards

1. Downloaded the pinned Betty snapshot and verified its inventory and hashes. Raw data remain excluded from Git.
2. Audited metadata, schemas and sampled field completeness. Separated advertised schema from usable observed fields.
3. Reviewed the motivating paper, public implementation and related literature. Recorded adaptation boundaries and ownership.
4. Defined chronological partitions by complete UTC dates, keeping every match together. Recorded the absence of series identifiers and recurring team limitation.
5. Corrected clock alignment using preceding combat event anchors. Mapped observed hero identities to metadata drafts instead of inferring side from replay slots.
6. Built the reference adaptation and comparison models. Fitted preprocessing only on training data and separated model validation from calibrator fitting.
7. Executed development explanations, uncertainty checks and the movement comparison. Deferred unsupported hero, item and vision extensions.
8. Completed integrity checks and froze settings, comparisons, calibration reporting and uncertainty summaries. Earlier invalid development diagnostics were withdrawn.
9. Evaluated the final set under that frozen protocol and produced the prespecified summaries. No further tuning followed final evaluation.
10. Saved result files and the literature evidence trail for the README, notebooks and IEEE style report.

## Reading the completed findings

Use `D_summary.json` and the underlying result files for numerical conclusions. Do not infer success from an expectation in this roadmap. Seed spread, sampling intervals and bootstrap intervals answer different questions. A small observed performance difference should be read alongside its prespecified paired uncertainty analysis.

The study cannot establish that changing a KPI causes winning, reproduce the original numerical result across different games, or generalise professional matches to every rank, region, team or patch. Negative and inconclusive findings are valid independent study outcomes.
