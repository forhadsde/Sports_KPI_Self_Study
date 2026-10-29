# Paper notes and adaptation boundaries

Author: Md Forhadul Islam

Updated: 29 October 2026

These notes connect the verified literature to the completed independent Dota 2 study. Detailed source access levels, metadata checks and claim support are in `docs/LITERATURE_REVIEW.md`, `outputs/results/D_literature_sources.json` and the Excel literature tracker. A paper checked only at abstract level is not treated as a verified source for detailed evaluation settings.

## Motivating paper and credit

Fazilat Hojaji, Robert E. McIlroy, Antoine Dupuy, Gianpaolo Pedroni, Adam J. Toth and Mark J. Campbell, “Deep learning techniques for identifying KPIs in League of Legends: Win prediction, map navigation, and vision control”, Computers in Human Behavior Reports, 2025. [Original paper](https://doi.org/10.1016/j.chbr.2025.100718).

The original method and findings belong to these authors. Their study combines outcome prediction and key performance indicator analysis with movement and vision analyses. Their [public implementation](https://github.com/ESRLAccount/LoL-Data-Analysis-and-Prediction) was inspected separately from the paper. It contains alternative model branches, so one branch is not automatically an exact record of the published experiment.

The original full dataset is described as available on request, with anonymised supplementary samples. Betty is a separate public dataset published by different owners. This project is an adaptation of selected workflow components, not a numerical reproduction of the paper or a reconstruction of its full project.

Section 5 identifies uneven rank representation, EU West only coverage, patch changes, sparse positional sampling, and champion classes, item builds and team composition as limitations or future directions. This section was verified from an accessible indexed copy of the author uploaded full text during the review. Live Chrome publisher and ResearchGate pages remained access blocked. Calibration and KPI ranking stability are supporting checks proposed for this independent study, not future work explicitly attributed to the authors.

## What was retained and what changed

| Component | Completed study decision |
|---|---|
| Outcome prediction | Adapted to Dota 2 using observed team feature differences |
| Reference neural network | Architecture and settings recorded in the frozen protocol; no claim of exact numerical equivalence |
| KPI interpretation | SHAP explanations and ranking summaries on development data |
| Game progression | Fixed observation times instead of claiming identical original phase definitions |
| Model comparison | Logistic regression, Random Forest, XGBoost and the reference network on the same partitions |
| Reliability | Raw and temperature adjusted probabilities, seed spread and bootstrap uncertainty |
| Movement | Separate paired sampling sensitivity experiment on eligible development trajectories |
| Champion, hero and item context | Deferred after the data audit |
| Vision control | Spatial ward reconstruction not claimed because sampled coordinates were absent |

The implemented neural network uses hidden layers of 64 and 64 units with dropout, Adam optimisation and a binary outcome objective. These settings come from `outputs/results/D_protocol.json`. They describe this implementation, rather than proving equality with the original architecture.

The observed inputs are net worth, experience, kills, assists, last hits, denies, level, health, living player count and living tower count, expressed as Radiant minus Dire. Their preparation is recorded in `D_preparation.json`. Eventual duration, final statistics, outcome columns, match identifiers and retrospective roles are excluded from prediction inputs.

## Existing Dota 2 and MOBA work

**Hodge and colleagues, mixed rank prediction.** The [author deposited preprint](https://arxiv.org/abs/1711.06498) compares logistic regression and Random Forest using draft and changing team statistics, and examines professional and highly skilled public matches. Its inspected methods include chronological mixed data evaluation and separate tournament evaluation. This establishes that baseline comparison and professional prediction already have precedents. The mixed rank preprint must not be conflated with the later journal paper titled Live Professional Match Prediction.

**Yang, Qin and Lei, real time prediction.** Their [paper](https://arxiv.org/abs/1701.03162) combines player and hero history with evolving economic and combat information. The inspected methods describe a match split and training cross validation. Its DOI issued year is 2017 despite an earlier submission timestamp. It motivates careful separation of information available before and during play, but does not by itself verify chronological evaluation for our study.

**Drachen and colleagues, spatial team behaviour.** Their [author deposited paper](https://arxiv.org/abs/1603.07738) describes zone changes, team dispersion and temporal clustering across skill levels. The abstract was inspected. Spatial analysis is therefore established; professional only Betty data cannot reproduce the original comparison across skill tiers.

**Eggert and colleagues, behavioural roles.** Their [role classification study](https://doi.org/10.1007/978-3-319-24589-8_9) supports treating roles as behavioural concepts rather than assuming every hero has a single fixed role. The publisher abstract was inspected. A role inferred from a whole match is not automatically a valid input for earlier prediction.

**Silva and Lopes, recent LoL attribution.** Their [publisher record](https://sol.sbc.org.br/index.php/sbgames/article/view/45416) describes composition, itemisation, spatial descriptors and time observable state estimation. Its abstract and bibliography were inspected. This overlaps broad temporal and contextual novelty claims, so the present study makes no claim to invent these ideas.

## Reliability and measurement literature

[Guo and colleagues](https://proceedings.mlr.press/v70/guo17a.html) provide an established basis for evaluating probability calibration and temperature scaling. Their evidence comes from other classification settings, so improvement on Dota 2 was a hypothesis rather than a promised outcome.

[Lundberg and Lee](https://arxiv.org/abs/1705.07874) introduce SHAP attribution. [Tree attribution work](https://arxiv.org/abs/1802.03888) supports explaining tree ensembles with suitable methods. These papers' theoretical consistency properties do not establish stability across training samples or random seeds.

[Aas and colleagues](https://doi.org/10.1016/j.artint.2021.103502) explain why feature dependence matters for attribution. Economic and combat measurements overlap, so an exact importance order should be interpreted cautiously. [Alvarez Melis and Jaakkola](https://arxiv.org/abs/1806.08049) motivate explanation robustness, while their input sensitivity question differs from the training resampling analysis used here.

[Noonan and colleagues](https://doi.org/10.1186/s40462-019-0177-1) provide a general measurement rationale for sampling sensitivity. Their animal tracking setting is not a direct model of game teleportation, death or respawn. Native sampled coordinates are a reference measurement, not continuous ground truth.

## Completed evaluation and interpretation

The frozen protocol records observation times of 600, 1200 and 1800 seconds and seeds 0, 1 and 2. Complete UTC dates are kept together in chronological training, validation, calibration and final test partitions. Series identifiers were unavailable, and recurring teams are not excluded. This is not an unseen team evaluation.

Snapshot clocks and player team identities were corrected before final evaluation. Earlier invalid development diagnostics were withdrawn. Integrity checks compare draft and outcome fields, protect temporal ordering and confirm partition consistency; they are consistency checks rather than independent ground truth. Final evaluation is complete without further tuning.

Observed results are in `D_final_metrics.json` and `D_summary.json`. They support comparing the implemented models, not claiming that one model is universally best. Explanation seed variation, fixed model bootstrap variation and training refit variation describe different uncertainties. The refit study is exploratory and uses development data. Common cohort evaluation helps distinguish changes in observation time from changes in eligible match populations.

The movement experiment compares matched continuous segments under native and coarse sampling with several grid offsets. It measures operational sampled path displacement, not true walking distance or a causal effect on winning. It does not establish how sampling changed model accuracy or reproduce the magnitude of any limitation in League of Legends.

Hero context remains deferred because retrospective role timing and generic item values prevented a defensible prospective extension. The study does not resolve the original paper's rank, region or patch limitations and makes no novelty or causal claim.

## Earlier Zenodo feasibility check

The League of Legends Zenodo source was considered before Dota 2 was selected. `outputs/results/A_data_feasibility.json` records an earlier inspection of an individual CSV. It supported a reduced prediction and explanation task but did not contain the full movement, vision and contextual information required for the complete original workflow. It was not used for the completed Dota 2 results. Statements about percentage snapshots remain retrospective and must not be interpreted as fixed minute live predictions.
