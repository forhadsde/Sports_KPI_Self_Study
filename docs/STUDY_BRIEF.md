# Independent study of KPI reliability in Dota 2

This brief preserves the original study intentions. The completed question by question answers, deferred extensions and evidence are in STUDY_ROADMAP.md and the executed notebooks. The final evaluation is complete; the future tense below records the initial plan rather than unfinished model training.

Author: Md Forhadul Islam, London

## Purpose and credit

This is an independent study motivated by Dr Fazilat Hojaji and her coauthors' work, Deep learning techniques for identifying KPIs in League of Legends: Win prediction, map navigation, and vision control. Their paper, method and findings are credited to Fazilat Hojaji, Robert E. McIlroy, Antoine Dupuy, Gianpaolo Pedroni, Adam J. Toth and Mark J. Campbell. Paper: https://doi.org/10.1016/j.chbr.2025.100718.

I will adapt the available parts of their approach to Dota 2 using the Betty Decision Context Dataset published by wolframko: https://huggingface.co/datasets/wolframko/betty-dota2. The publisher describes a pipeline using Valve replays and STRATZ data. The dataset and its collection are not my contribution. This is a transfer study, rather than an exact reproduction of their League of Legends results. No affiliation with their research group is claimed.

## The problem

Predicting a winning team and identifying a reliable performance indicator are different tasks. A model can predict well while its feature rankings vary with the model, the sampled matches or the measurement interval. Movement measurements also depend on how often positions are recorded. Hero roles and team composition may change what a statistic means.

The aim is to investigate how reproducible the predictions and explanations are when the authors' general workflow is applied to a different team game. Indicators are predictive associations, not evidence that changing one statistic causes a win.

## Agreed core and possible extensions

The original agreed core asks whether model families differ in predictive performance, whether their probabilities are calibrated, and whether KPI rankings remain stable across seeds and samples. The feature group experiment was removed at Forhad's request. These questions remain sufficient for an independent study. Movement sampling was subsequently proposed as a focused extension connected to the authors' limitations, and hero context remains conditional. They should not be described as part of an exact reproduction or as confirmed novel research gaps.

## Research questions and reasons

### RQ1: Can an adaptation of their approach identify predictive KPIs in Dota 2?

Construct team statistics from the available data and compare a neural network inspired by their reported architecture with logistic regression, Random Forest and XGBoost using identical partitions. Explain differences in inputs and preprocessing explicitly. Evaluate discrimination, classification performance and calibration. Use fixed observation times and only information available by each observation time for prospective prediction. An end of match analysis, if included, is a separate retrospective task.

Why: this establishes a transparent reference before investigating limitations. It does not test whether the authors' numerical result can be reproduced across different games.

### RQ2: How much does movement sampling resolution change the measurements and conclusions?

Use the same trajectories and observation windows at their native sampling interval and after downsampling to minute intervals. Compare path length, movement summaries and any resulting KPI rankings. Repeat coarse sampling with several time offsets to avoid making the conclusion depend on one sampling grid. Do not interpolate across death, respawn, teleportation, missing intervals or discontinuities without explicit rules. Native sampled trajectories are a reference measurement, not continuous ground truth.

Why: Section 5 of Hojaji and colleagues' paper identifies sparse position sampling as a limitation. This experiment quantifies a related measurement issue in Dota 2. If longer step intervals underestimate distance, quantify the size and variability rather than presenting the direction alone as a new discovery.

### RQ3: How stable are KPI rankings across samples, training seeds and model families?

Use match level bootstrap samples and repeated training seeds. Separate uncertainty from reusing a fixed fitted model's explanations from uncertainty caused by refitting a model on different training samples. Report rank agreement, leading feature inclusion frequency and uncertainty intervals. Use suitable explanation methods for each model and acknowledge correlated features.

Why: this assesses how strongly an independent reader can trust an explanation. This is a supporting question proposed for this study, not a future work claim attributed to the original authors.

### RQ4: Does hero context change the interpretation of the KPIs?

Conditional extension after the audit: use documented lane roles or defensible hero role labels to compare explanations, or test whether hero draft context changes predictions. Item features must reflect inventory or purchases already observed by the prediction time. Avoid defining a pregame subgroup using later match outcomes or final items. Check subgroup sample sizes and keep the number of comparisons modest.

Why: the authors explicitly identify champion classes, item builds and composition as directions for further research. Dota 2 has related concepts but different mechanics. A single hero can fill multiple roles, so a fixed class label is not automatically valid.

## Evaluation and limits

Keep every observation from a match in the same partition. Inspect dates, series, teams and player identifiers before defining the split. Prefer chronological partitions and group series where identifiers permit it. If teams appear in multiple chronological partitions, report that fact rather than claiming unseen team generalisation. Fit preprocessing on training data only. Use development data for model settings, calibration and experimental decisions. Freeze the protocol before final test evaluation. Reusing frozen test predictions for prespecified uncertainty summaries is part of that final evaluation, not permission to tune models on test results.

Use the seeds specified in src/common.py. Save measurements, settings, exclusions and results under outputs/results. Preserve a separate final test set and do not inspect its labels or outcomes to decide which method to report. Exclude final scores, final economy, end of match damage, eventual duration and other future information from earlier prediction inputs. Models at later observation times apply only to matches still active then.

Regional and rank generalisation are not promised. Professional matches do not represent the full ranked player population. Patch comparisons require verified metadata and sufficient coverage. The dataset card describes fields that still need inspection for completeness, parser artefacts and join consistency. No claim of resolving all limitations of the original paper or of research novelty is made.

## Deliverables

Executed Jupyter notebooks with saved outputs, shared Python code, a documented environment, a short IEEE style report and a README that links all results to generated files. Forhad is the author of this independent study. Original research and data ownership remain clearly credited. Rugby and video detection are deferred. Publication and sending any email require Forhad's confirmation.

## Terminology clarification

Applying the workflow to Dota 2 is an adaptation or transfer study. It does not reproduce the original League of Legends result. A reference implementation within this project establishes an internal baseline, not a numerical reproduction of the paper.

The gap addressed by the original paper was the limited integration of win prediction with movement, map control and vision analysis in earlier work. The authors' stated remaining limitations and future directions include uneven rank representation, regional coverage, game updates, coarse position sampling and champion or item context. Our Dota 2 sampling experiment investigates a related measurement issue in another game. Calibration and ranking stability are supporting checks proposed for this independent study, not future directions explicitly attributed to the original authors. Professional Dota 2 data cannot directly resolve the paper's uneven coverage of ordinary player ranks.
