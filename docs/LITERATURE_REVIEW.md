# Literature review for the independent Dota 2 study

Author: Md Forhadul Islam

Review date: 29 October 2026

## Purpose and scope

This focused, critical review informs the experimental protocol before training. It covers the approach of Dr Fazilat Hojaji and her coauthors, related multiplayer online battle arena research, probability calibration, explanation reliability and movement measurement. It is a structured narrative review, not an exhaustive systematic review. Discovery searches used Google in Google Chrome, followed by primary publisher, author repository, arXiv and proceedings records. Google Scholar was opened as an initial entry page, but no completed Scholar search is recorded. Search result summaries and generated search overviews were not accepted as evidence for paper findings.

Search phrases included Dota 2 win prediction machine learning, Skill based differences Dota, Classification of Player Roles Dota, Hojaji KPI limitations, Interpretable Real time Win Prediction, movement sampling frequency distance travelled Noonan, and robustness interpretability methods Alvarez Melis Jaakkola. Reference lists in accessible papers supplied further candidates. Selection prioritised papers directly affecting the planned questions and evaluation. A source appearing in another paper's references was not treated as independently verified.

The evidence ledger is `outputs/results/D_literature_sources.json`. Bibliography keys match `report/references.bib`. Abstract only means that the abstract and bibliographic record were inspected, but experimental implementation details were not independently established. Full text selections means the accessible article and relevant methods passages were inspected, not that every derivation or supplementary file was checked. A second metadata record was checked for every paper where available. Crossref exposes publisher deposited metadata and DataCite exposes arXiv deposited metadata. Agreement between two records does not constitute independent editorial confirmation. Dataset provenance checks remain incomplete where the publisher linked repository was unavailable.

## Terms in plain language

* A KPI, or key performance indicator, is a measurement used to describe performance. Here it is a candidate predictive indicator, not automatically an actionable coaching instruction.
* Discrimination is the ability to rank eventual winners above losers. ROC AUC summarises this ranking ability across decision thresholds.
* Calibration asks whether predicted probabilities match observed frequencies. Among comparable predictions near a given probability, the observed fraction of wins should be similar.
* SHAP assigns contributions to input features for a particular model prediction relative to a reference distribution. It explains a model under specified assumptions; it does not prove causation.
* Correlated features carry overlapping information. Gold, experience, kills and objectives can move together, so a model may distribute importance among them differently.
* Bootstrap uncertainty comes from repeatedly resampling independent units with replacement. For this study the unit is the match, or a larger series cluster where appropriate, rather than individual frames.
* Leakage means using information unavailable at prediction time, or allowing related observations to cross training and evaluation boundaries in a way that makes evaluation optimistic.
* An adaptation transfers a workflow to another setting. Reproducing the original numerical result would require a sufficiently equivalent task, data and protocol.

## Evidence matrix

| Key and primary source | Evidence inspected | What it supports | Limits and consequences for this study |
|---|---|---|---|
| Hojaji2025, [original paper](https://doi.org/10.1016/j.chbr.2025.100718) and [researchers' code](https://github.com/ESRLAccount/LoL-Data-Analysis-and-Prediction) | Repository and model file inspected in Chrome. Section 5 verified from an accessible indexed author uploaded full text during this review. Crossref metadata checked | Prediction, SHAP and movement analysis; stated limitations include ranks, region, patches, sampling and contextual features | Chrome live paper pages remained access blocked. Published architecture and implementation branch correspondence are documented separately |
| Hodge2017, [mixed rank prediction](https://arxiv.org/abs/1711.06498) | Abstract and full text methods and discussion | Logistic regression and Random Forest, hero vectors and time window statistics, mixed and professional populations | This is the mixed rank preprint. Do not conflate it with the later paper titled Live Professional Match Prediction |
| Yang2016, [real time prediction](https://arxiv.org/abs/1701.03162) | Abstract and full text methods selections | Prior player and hero information, gold, experience and deaths over time, logistic regression and Attribute Sequence Model | Match split and training cross validation are described. Chronological ordering is not established by the inspected passage |
| Akhmedov2021, [outcomes prediction](https://arxiv.org/abs/2106.01782) | Abstract and full text selections | A Game State Integration collection tool and comparison of regression, neural network and LSTM approaches | The target and multi forward forecasting task must be checked before comparing headline performance with binary win prediction |
| Drachen2014, [spatial behaviour](https://arxiv.org/abs/1603.07738) | Author deposited abstract and journal reference | Zone changes, team dispersion and temporal clustering already relate spatial behaviour to skill tiers in Dota 2 | Skill association is different from prospective win prediction. Professional only Betty data cannot reproduce rank comparisons |
| Eggert2015, [role classification](https://doi.org/10.1007/978-3-319-24589-8_9) | Publisher abstract, author list, venue and citation | Player roles can be learnt from replay derived behavioural attributes; roles are not formally unambiguous labels | A role inferred from later behaviour can leak future information when used as an earlier prediction input |
| Yang2020, [interpretable Honor of Kings prediction](https://arxiv.org/abs/2008.06313) | Author deposited abstract and version metadata | Interpretable spatial and temporal MOBA prediction predates this study | Different game, proprietary collection and architecture. Exact split details not verified here |
| Silva2026, [temporal and spatial LoL attribution](https://sol.sbc.org.br/index.php/sbgames/article/view/45416) | Publisher abstract, references and bibliographic record | Recent work already combines composition, itemisation, spatial descriptors, time observable states and forecasting, and cites Hojaji and colleagues | Major overlap with broad proposed novelty claims. The abstract's use of causal sequence must not be read as established intervention causality. Full evaluation details remain unverified |
| Guo2017, [neural calibration](https://proceedings.mlr.press/v70/guo17a.html) | Proceedings abstract and publisher BibTeX | Neural network confidence can be miscalibrated; temperature scaling is an established adjustment | Experiments concern image and document datasets. Improvement on Dota 2 is a hypothesis, not a guaranteed result |
| Lundberg2017, [SHAP framework](https://arxiv.org/abs/1705.07874) | Author deposited abstract and version record | Additive local feature attribution and its theoretical motivation | The framework's consistency property is different from stability across data samples or training seeds |
| Lundberg2018, [tree explanations](https://arxiv.org/abs/1802.03888) | Author deposited abstract and version record | Exact efficient tree SHAP methods and interaction attribution | Explain Random Forest and boosted trees with appropriate methods; do not equate an arbitrary network weight with SHAP |
| Aas2019, [dependent features](https://arxiv.org/abs/1903.10464) | Author deposited abstract and version record | Independence assumptions can distort explanations; conditional approximations and grouped explanations address a different question | Record explanation assumptions and background data. Grouped reporting is an explanation summary, not the removed feature group ablation experiment |
| AlvarezMelis2018, [explanation robustness](https://arxiv.org/abs/1806.08049) | Workshop preprint abstract and venue note | Similar inputs can receive unstable explanations, motivating robustness assessment | Input perturbation robustness is different from refitting on bootstrap training samples. Do not claim this paper implements our exact ranking protocol |
| Noonan2019, [movement sampling](https://doi.org/10.1186/s40462-019-0177-1) | PubMed abstract and publisher metadata linkage | Straight line distance is sampling sensitive; frequency, tortuosity and measurement error affect estimates | Animal tracking is supporting measurement literature. Its continuous movement assumptions do not automatically fit teleportation, death and respawn in games |
| Breiman2001, [Random Forests](https://doi.org/10.1023/A:1010933404324) | Publisher abstract and full citation | Original ensemble method and variable importance | Built in importance is not automatically interchangeable with SHAP |
| Chen2016, [XGBoost](https://arxiv.org/abs/1603.02754) | Author deposited abstract and related publisher DOI | Original tree boosting system | Cite the original method as well as recording the installed software version |

## What is established already

Dota 2 outcome modelling, comparison of simpler and more complex models, and hero context are established topics. Hodge and colleagues compare logistic regression and Random Forest using draft and evolving team statistics. Yang, Qin and Lei combine player history and hero information with changing economic and combat measurements. Their full text explicitly distinguishes prior information from information observed during play. Consequently, applying a neural network or SHAP to Dota 2 is a useful independent implementation exercise, but cannot by itself justify a new method claim.

Spatial behaviour is also established. Drachen and colleagues analyse zone changes and team distances across skill tiers. Eggert and colleagues show that roles can be inferred from behavioural attributes. These studies warn against interpreting one universal ranking as equally meaningful for every role or skill population. Their results do not establish that positioning causes victory.

Recent LoL work requires particular care. Silva and Lopes published a temporal and spatial attribution framework before this review date. Its abstract includes composition, itemisation and spatial descriptors, with information restricted to that observable by each minute. It cites Hojaji and colleagues and Guo and colleagues. Therefore the broad statement that earlier research only predicts winners and never studies context or spatial attribution would be inaccurate. This source needs full methods review before final report submission.

## Original study and adaptation boundary

Credit for the motivating method and original findings belongs to Fazilat Hojaji, Robert E. McIlroy, Antoine Dupuy, Gianpaolo Pedroni, Adam J. Toth and Mark J. Campbell. The project adapts their workflow to another game. It cannot reproduce their League of Legends accuracy or validate their specific champion, bounty or vision findings using Dota 2.

The public `4-Deep learning_MatchOutcome.py` file contains several alternative implementations, so it must not be treated as a single unambiguous experimental specification. The inspected PyTorch timeline branch uses an active hidden layer, dropout, a rectified linear activation, a binary classification loss, Adam optimisation and SHAP DeepExplainer. It also drops match identifiers before an ordinary random row split, observes the named test partition during training, and normalises timeline quantities using eventual duration. These are observations about this public branch, not proof of what the final published experiment ran. They justify documenting adaptations explicitly and building an evaluation with separate development and final test partitions. Eventual duration must not enter an early prediction feature or its normalisation.

Section 5 identifies uneven rank coverage, EU West only data, game patch changes, minute interval positions, champion classes, items and composition as limitations or future directions. This section was verified through an accessible indexed copy of the author uploaded paper during the review. The live Chrome publisher and ResearchGate pages remained blocked. Calibration and explanation ranking stability are our proposed supporting checks, rather than future directions explicitly requested by the authors.

## Defensible study questions and expectations

### Reference adaptation

**A1. Can a documented adaptation predict Dota 2 outcomes at fixed observation times?** Compare a reference network with logistic regression, Random Forest and XGBoost using the same eligible matches and inputs. Existing research makes predictive signal plausible, but neither model superiority nor a numerical target is assumed. This establishes whether the pipeline works under a transparent evaluation.

**A2. Which observed statistics contribute to its predictions?** Use appropriate feature attribution methods with declared background samples and output scale. Economic and objective measurements are plausible indicators, but the ranking remains unknown. This adapts the prediction and explanation workflow without claiming a new explanation algorithm.

**A3. Do explanations change as a match develops?** Compare prespecified observation times. Rankings may change with game state, but later times include only matches still active. Report eligible populations and, where useful, a common cohort sensitivity analysis. This is a descriptive temporal analysis already motivated by previous work.

### Reliability checks after the reference adaptation

**B1. Does model choice change predictions and explanations?** Compare paired results on identical matches. Simpler models may be competitive, and similar predictions may accompany different rankings. This tests dependence on modelling choices rather than assuming a neural network is necessary.

**B2. Do predicted probabilities reflect observed outcomes?** Report Brier score, log loss and reliability diagrams with sample counts. Brier score measures squared probability error and reflects more than calibration alone; log loss penalises confident errors. Develop temperature scaling or another justified calibrator only on development data. Calibration is an established evaluation task whose value here is trustworthy probability reporting.

**B3. Are KPI rankings stable?** Separate training seed variation, refitting variation under match resampling, and explanation variation for a fixed model. Report rank agreement and leading feature inclusion frequency. Correlated measurements may exchange importance, so inspect dependence and interpret groups cautiously. SHAP consistency does not imply this type of stability.

### Selected measurement experiment

**C1. How does trajectory sampling change movement estimates and their interpretation?** Compare native sampled trajectories with coarser versions of the same trajectories and windows. Repeat coarse grids with different offsets. Measure paired differences in distance, dispersion or zone transitions and assess whether substantive rankings or associations change. Missing intermediate turns can reduce polygonal distance under nested sampling with identical endpoints, so the direction alone is not a novel discovery. The interesting result is the magnitude, variation and effect on conclusions under explicitly defined game specific rules.

Treat native sampling as a reference, not continuous ground truth. Exclude or separately analyse death, respawn, teleportation, large jumps and missing stretches. Do not claim a speed threshold identifies every teleport. Distinguish location changes from physical walking distance. Coarse and fine segment eligibility must be comparable, otherwise the comparison can reflect different filtering rather than sampling alone. Use match level paired uncertainty summaries.

### Conditional context extension

**C2. Does a small, justified hero or role context change interpretation?** Only proceed if labels are meaningful, sufficiently populated and available at the relevant prediction time. A publisher's role label may be inferred after the match. Current inventory and purchase events also need a timing and identity audit. Differences between subgroups can reflect confounding, selection and sample size. This is an exploratory contextual analysis, not a causal estimate or a comprehensive item recommendation system.

## Dataset and attribution constraints

The Betty card credits wolframko, Valve replays, STRATZ enrichment and the manta parser. It declares MIT and documents a rich schema. The card is a source description, not validation that every advertised field is usable. Its own aggregate table and viewer describe different row totals, illustrating why audit counts must come from pinned files. Local audit findings already indicate missing ward coordinates, generic item strings and empty role labels in inspected samples. Those limitations override optimistic schema descriptions when choosing experiments. Regional, rank and patch generalisation remain conditional on actual metadata and coverage.

The dataset card should be cited separately from the methods papers. Record installed software versions and cite original method papers for SHAP, Random Forest and XGBoost when those methods are used. A final report bibliography audit should add verified software references and reconcile every citation with its use. The current bibliography is a verified starting set, not a claim that every eventual software dependency is already cited.

### Implementation credits and current audit decisions

The bibliography and tracker also credit Pedregosa and colleagues' scikit learn paper and Paszke and colleagues' PyTorch paper. The journal author list is retained for scikit learn because a later arXiv revision adds authors. PyTorch uses the official proceedings author list. These are software credits, not claimed contributions of this independent study. Versions and settings still need recording with the experiments.

The hero identity join uses the OpenDota Project and contributors' dotaconstants dictionary at a pinned revision. Its MIT licence and dictionary fields were checked separately in Chrome. This is a documentation resource without an abstract. Its provenance is recorded in `outputs/results/D_hero_constants_source.json`. The dictionary maps internal hero names to numeric identifiers; it does not independently establish which team a replay slot belongs to.

The development data audit found that replay game time and combat log timestamps can differ because of pauses. Corrected snapshot construction aligns raw ticks with safe preceding combat timestamp anchors. Replay slot order also does not reliably indicate Radiant or Dire. Team assignment therefore joins observed hero names, through the pinned dictionary, to metadata drafts and requires distinct heroes with the expected team membership. Blank hero names are excluded. These are implementation audit findings, not claims made by the literature. Earlier invalid development diagnostics were withdrawn to local excluded cache and are not retained as findings. Corrected development analyses and integrity checks preceded the protocol freeze and final evaluation. The final set was protected during development, and final evaluation has now completed without further tuning. Integrity checks and their limits are recorded separately in `outputs/results/D_integrity_validation.json`.

## Recommended order and decision gates

1. Complete pinned download and file integrity checks, then audit actual fields and clocks.
2. Record the original method and phase definitions from the available full text, then compare them with public code. Record differences instead of silently copying a branch. Section 5 limitations have been verified.
3. Freeze match and preferably series grouped chronological partitions. State recurring team overlap. Fit preprocessing only on training data.
4. Build the reference adaptation and simple baselines. Keep all data from one match together.
5. Assess discrimination, calibration and ranking stability on development partitions. Separate calibrator fitting from its evaluation.
6. Run the paired sampling experiment with documented discontinuity rules and sampling offsets.
7. Proceed with context analysis only if its audit gate passes. Do not restore the removed feature group ablation.
8. Freeze the protocol, then evaluate once on final test data. Bootstrap frozen test predictions only for prespecified summaries, without tuning.
9. Write the short IEEE style report around observed evidence, uncertainty, credit and limitations.

## Related work paragraph for later report drafting

Hojaji and colleagues motivate the present independent adaptation through their integration of outcome prediction and KPI interpretation in League of Legends. Dota 2 prediction already includes hero and player history models, real time economic and combat measurements, and comparisons of logistic regression and tree ensembles. Spatial team behaviour and behavioural player roles have also been studied independently. Recent League of Legends work further combines temporal state estimation, composition, itemisation and spatial attribution. Accordingly, this study does not introduce win prediction or contextual explanation. It examines the reproducibility of a documented transferred workflow through probability calibration, explanation ranking uncertainty and movement sampling sensitivity. SHAP provides model attribution rather than intervention effects, while research on dependent features and explanation robustness motivates cautious interpretation. Sampling research supplies a general measurement rationale, with game discontinuities requiring separate treatment. Every claimed result will depend on the completed dataset audit and prespecified evaluation.

## Remaining review limitations

Most methodological sources in the matrix were checked at abstract level. Their specific experimental details must not be invented in the report. Original Section 5 statements are verified, while detailed comparison of the published architecture with public branches remains a separate implementation task. No exhaustive citation network search or formal novelty claim has been completed. Item specific Dota 2 prediction, direct ward spatial analysis and exact temporal split practices in several candidate papers need further primary full text reading if those become central experiments. The report should prioritise a smaller number of directly supported citations over an inflated bibliography.

## Bibliographic audit notes

The issued year of Real time eSports Match Result Prediction is 2017 in its DOI record, despite a December 2016 initial submission. The stable key Yang2016 is retained while the bibliography uses the issued year. Aas and colleagues' matching journal version was verified as Artificial Intelligence, published in 2021, and replaces the preprint metadata under the stable key Aas2019. The Honor of Kings journal version has a different author list from the inspected preprint, so the preprint identity is retained. Hodge's mixed rank preprint is also kept separate from the later live professional journal article. Drachen's author deposited name Klabjan is retained; the Crossref deposit spells it Klabajan. These distinctions prevent silently combining metadata from different versions. The dataset card's linked code branch returned a GitHub page not found response, so its source code provenance was not independently confirmed.
