"""Build the public overview with measured tables from the saved evidence."""
from pathlib import Path  # Use a portable project location.
import json  # Read the measured results rather than retyping them.
ROOT = Path(__file__).resolve().parents[1]  # Find the repository containing this script.
s = json.loads((ROOT / "outputs/results/D_summary.json").read_text())  # Load final results and limitations.
names = {"logistic": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost", "neural_network": "Neural adaptation"}  # Give model identifiers readable labels.
rows = [r for r in s["test_table"] if r["horizon_seconds"] == 1200 and r["probability"] == "raw"]  # Select the prespecified primary comparison table.
table = '\n'.join(f"| {names[r['model']]} | {r['auc_mean']:.4f} ± {r['auc_sd']:.4f} | {r['accuracy_mean']:.4f} | {r['brier_mean']:.4f} | {r['log_loss_mean']:.4f} |" for r in rows)  # Format direct measurements with seed variation.
p = s['primary_paired_comparison']  # Keep paired prediction uncertainty separate from seed metric means.
text = f'''# Independent study of KPI reliability in Dota 2

**Author: Md Forhadul Islam, London**

I adapted the win prediction and KPI explanation approach of **Dr Fazilat Hojaji and her coauthors** to public Dota 2 data. This independent study helped me practise statistics, data auditing and reproducible analysis. It is a transfer to a different game and population, **not an exact numerical reproduction** of their League of Legends results. I do not claim a new prediction algorithm or affiliation with the original research team.

The completed experiments compare models, probability calibration and explanation stability, and examine movement sampling as a separate measurement problem. Rugby, video detection, vision control and the hero context extension are outside the completed scope. The report remains local for my review.

## Follow the methodology in slides

The [editable study presentation](outputs/slides/Dota2_KPI_Methodology_and_Questions.pptx) explains the questions, data cleaning, statistical measures, model choices, training procedure, documented problems and fixes. Each question or major step has its own slide with figures and bullet points. The neural architecture and the reason for using a fixed reference width are explained separately.

The fitted architecture description and an additional source null metadata audit were added when explaining the methodology. They read saved weights and source headers. They did not retrain models, alter the original experiment or repeat final evaluation. Their evidence is saved in `outputs/results/D_neural_architecture_audit.json` and `D_poststudy_missingness_audit.json`.

## Start with the notebooks

Each notebook has saved outputs, plain language explanations before code, inline comments and interpretation afterwards. The teaching notebooks read the frozen experiment evidence; rerunning them does not reopen final model selection. Training and extraction implementations are in `src/`.

| Notebook | What I learn |
|---|---|
| [00 Data audit](notebooks/00_dota_data_audit.ipynb) | What was downloaded and what is actually usable |
| [01 Environment and literature](notebooks/01_environment_and_literature.ipynb) | Hardware, study questions, source review and credit |
| [02 Preparation](notebooks/02_data_preparation.ipynb) | Features, match clocks, team identity and chronological splits |
| [03 Model comparison](notebooks/03_model_comparison.ipynb) | Neural adaptation, baselines, metrics and uncertainty |
| [04 Calibration and KPI stability](notebooks/04_calibration_and_kpi_stability.ipynb) | Probability quality, SHAP and training resample stability |
| [05 Movement sampling](notebooks/05_movement_sampling.ipynb) | Matched path measurements and their limits |
| [06 Findings](notebooks/06_findings_and_next_steps.ipynb) | Answers, remaining questions and reproduction sequence |

## Questions and observed answers

1. Can the neural workflow be adapted to Dota? Yes, on the audited current team statistics. Different inputs and mechanics prevent a direct numerical comparison with LoL.
2. Does it outperform simpler models under identical splits? Logistic regression has a small primary advantage in this fixed comparison. This is not an exhaustive tuning study.
3. Are predicted probabilities trustworthy? I report Brier score, log loss and reliability curves before and after temperature scaling. Adjustment does not improve every model and metric.
4. Are KPIs stable? Net worth is prominent across models and actual training refits. Several other rankings change with the estimator and sampled training games. These are predictive associations, not causal instructions.
5. Does minute sampling change movement measurement? It retains substantially less sampled path length on qualifying matched continuous segments. Movement was not added to the prediction model, so no downstream accuracy improvement is claimed.

The full [roadmap](docs/STUDY_ROADMAP.md) explains each question, its reason, historical expectation and evidence. The [paper notes](docs/PAPER_NOTES.md) distinguish the original authors' limitations from my independent checks.

## Results

The primary time is 20 minutes, with **{p['matches']} eligible final matches**. Entries are means over the fixed training seeds; AUC also shows sample standard deviation. Higher AUC and accuracy are better. Lower Brier score and log loss are better. Raw probabilities are reported here.

| Model | ROC AUC, mean ± SD | Accuracy | Brier score | Log loss |
|---|---|---|---|---|
{table}

For seed averaged predictions, the paired logistic minus neural AUC difference is {p['observed_logistic_minus_neural']['auc']:.4f}. Its match bootstrap interval is [{p['match']['auc'][0]:.4f}, {p['match']['auc'][1]:.4f}], and its UTC day block interval is [{p['utc_day_block']['auc'][0]:.4f}, {p['utc_day_block']['auc'][1]:.4f}]. These conditional intervals favour logistic regression; the accuracy difference interval includes zero. The AUC of averaged predictions need not equal the mean of individual seed AUC values.

![Prediction performance](outputs/figures/D_performance.png)

Later horizons contain fewer eligible games. The notebooks also report a common match cohort to avoid silently changing membership when discussing time trends.

![Calibration](outputs/figures/D_calibration.png)

Temperature scaling was fitted on a separate chronological calibration period. Raw and adjusted results were both specified for reporting before final evaluation. Bin counts are saved with the curves; sparse bins can be noisy.

![KPI stability](outputs/figures/D_rank_stability.png)

SHAP explains fitted predictions relative to a training background. Explanation units differ across model families, so cross model comparisons use ranks. Fixed model explanation resampling, training seed spread and actual bootstrap training refits describe different uncertainties.

![Movement measurement](outputs/figures/D_movement.png)

Minute endpoint chords are compared with native sampled paths over identical segment endpoints. Deaths, missing observations and implausible jumps split segments. The direction follows the triangle inequality; the empirical result is the magnitude under these rules. Native positions are not continuous ground truth, and coordinates are not claimed to be metres.

## Data and evaluation safeguards

The Betty Decision Context Dataset is published by **wolframko**, using **Valve replays and STRATZ data**. Downloads are pinned and checked against source hashes. Raw and processed data remain local. See [data sources](docs/DATA_SOURCES.md) for the revision, download record and publisher declared licence caveat.

The audit found that replay elapsed time is not a reliable match clock and slot order does not reliably encode teams. Combat event anchors now define observation times, with strictly preceding states and a conservative combined timing bound of 2.4 seconds. Hero identities map to metadata drafts using the pinned OpenDota dictionary. The historical `max_snapshot_age_seconds` field records the state matching component alone; `snapshot_age_bound_seconds` is the combined bound.

Whole matches stay together in chronological train, validation, calibration and final periods, grouping complete UTC dates. Every model receives identical eligible matches and features. The final analysis protocol was frozen before the single final evaluation. The evaluator refuses a repeated final run. Same teams can recur and series identifiers are unavailable, so this does not establish generalisation to unseen teams or tournaments.

The predictor uses net worth, experience, kills, assists, last hits, denies, level, health, alive players and alive towers, each as Radiant minus Dire totals. It excludes future statistics, final duration as a predictor, identifiers and retrospective roles. The proposed feature group ablation is not part of the completed experiments. Ward locations and detailed inventory provenance do not support the deferred extensions.

## Reproduce locally on Windows

Use Anaconda and a CUDA capable NVIDIA GPU for the full experiment. The recorded environment used the RTX 3090 for PyTorch and XGBoost. A reader can inspect the notebooks and compact evidence without downloading raw data.

```powershell
# Create an environment within this project.
conda env create --prefix ./.conda --file environment.yml
# Expose the Windows numerical library directories, including paths containing spaces.
$env:PATH = "$PWD/.conda;$PWD/.conda/Library/bin;$PWD/.conda/Scripts;" + $env:PATH
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
# Start Jupyter with this environment.
./scripts/start_jupyter.ps1
```

To read or refresh saved notebook outputs without training:

```powershell
./.conda/python.exe scripts/execute_notebooks.py
```

To reproduce the scientific pipeline on a **separate fresh copy**, follow this order. Existing published final evidence intentionally prevents overwriting the original evaluation. Do not delete the study's historical results to tune against its final outcomes. A fresh experiment needs its own results directory or clean working copy without the completed experiment evidence.

```powershell
# Obtain the pinned raw tables and hero dictionary with licence provenance.
./.conda/python.exe src/download_dota2.py
./.conda/python.exe src/download_hero_constants.py
# Audit source tables and extract conservatively timed features.
./.conda/python.exe src/audit_dota2.py
./.conda/python.exe -c "import sys; sys.path.insert(0,'src'); from dota_data import prepare; prepare()"
./.conda/python.exe src/validate_dota_data.py
./.conda/python.exe -m unittest discover -s tests
# Fit only development models, then calculate development explanations and movement measures.
./.conda/python.exe src/run_dota_study.py --stage development
./.conda/python.exe src/dota_explanations.py
./.conda/python.exe src/dota_movement.py
# Review development evidence, record decisions, then evaluate the final period once.
./.conda/python.exe src/freeze_dota_protocol.py --development-reviewed
./.conda/python.exe src/run_dota_study.py --stage final
# Summarise saved predictions and produce figures without refitting.
./.conda/python.exe src/summarise_dota.py
./.conda/python.exe src/plot_dota_results.py
```

Source comments explain implementation choices and assumptions. The freeze file retains the original implementation fingerprints. Later teaching and packaging changes are documented separately; frozen fingerprints are not retroactively replaced. A future scientific extension needs a new untouched test period.

## Evidence and literature

* [Literature review](docs/LITERATURE_REVIEW.md) and [Excel review tracker](outputs/literature/Literature_Review_Tracker.xlsx), including short abstract summaries, searches, access limits and claim links.
* `outputs/results/D_summary.json` supplies result tables, paired intervals and interpretation limits.
* `D_protocol.json`, `D_integrity_validation.json` and `D_download_manifest.json` record protocol, checks and acquisition provenance.
* Per seed metrics, SHAP summaries, training refit records, movement pairs and probability bin counts are retained in `outputs/results/`.
* Compact final and development predictions in `outputs/predictions/` support replotting without model caches.
* Raw data, processed tables, trained models and the draft report are excluded from this public repository.

## Limits and next steps

This observational study is conditional on the available professional matches, audited fields, fixed settings and finite samples. Chronology cannot eliminate team or tournament dependence. Correlated predictors complicate attribution. A stable explanation is not a causal recommendation. Continuity filtering restricts movement coverage. Hero roles, item context, ward geography, rugby transfer and movement prediction remain unanswered. New analyses require a prospectively defined protocol and a new evaluation period.

## Credit

The motivating method and original findings belong to **Fazilat Hojaji, Robert E. McIlroy, Antoine Dupuy, Gianpaolo Pedroni, Adam J. Toth and Mark J. Campbell**, [Deep learning techniques for identifying KPIs in League of Legends: Win prediction, map navigation, and vision control](https://doi.org/10.1016/j.chbr.2025.100718), Computers in Human Behavior Reports, 2025. Their [public implementation](https://github.com/ESRLAccount/LoL-Data-Analysis-and-Prediction) informed method inspection; this project independently implements its Dota adaptation.

Dataset credit belongs to [wolframko and the credited upstream providers](https://huggingface.co/datasets/wolframko/betty-dota2). Identity constants come from [OpenDota](https://github.com/odota/dotaconstants). SHAP, scikit learn, XGBoost, PyTorch, NumPy, pandas, SciPy, PyArrow, Matplotlib and Jupyter provide the research software. Their primary references and the wider literature are recorded in the source tracker. The original method, public data and software are not my work; the independent implementation, audit and study write up are my contribution.
'''
(ROOT / "README.md").write_text(text, encoding="utf-8")  # Save the public overview with measured values.
print("README generated from measured results.")  # Report completion without recalculating any experiment.
