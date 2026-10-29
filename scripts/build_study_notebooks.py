"""Write the teaching notebooks from saved evidence, without reopening test selection.

The generated notebooks are ordinary editable Jupyter files. Shared scientific
implementations live in src so that explanations and computations stay separate.
"""
from pathlib import Path  # Locate the project independently of the terminal folder.
import nbformat as nb  # Create valid notebook cells rather than hand editing JSON.

ROOT = Path(__file__).resolve().parents[1]  # Resolve the project containing this script.
INTRO = """Md Forhadul Islam, London

This is my independent Dota 2 adaptation of the approach of Fazilat Hojaji,
Robert E. McIlroy, Antoine Dupuy, Gianpaolo Pedroni, Adam J. Toth and Mark J. Campbell.
Their original League of Legends paper is https://doi.org/10.1016/j.chbr.2025.100718.
It is not an exact reproduction of their numerical results. The Betty dataset is
published by wolframko using Valve replays and STRATZ data. Sources, licences and
the literature tracker are recorded in the project documentation.

I explain each step before the code and interpret the output afterwards. All
displayed experimental results come from saved calculations. Reading them does
not train a new model or choose an approach using the final test outcomes.
"""
SETUP = '''from pathlib import Path  # Represent folders consistently on Windows and other systems.
import sys  # Make the shared project modules available to this notebook.
import json  # Read saved evidence with its original field names.
import pandas as pd  # Display numerical records as labelled tables.
from IPython.display import display, Image, Markdown  # Show readable tables, figures and notes.
ROOT = Path.cwd().resolve()  # Begin from the notebook execution folder.
if ROOT.name == "notebooks":  # Jupyter normally starts within the notebook folder.
    ROOT = ROOT.parent  # Move up once to reach the shared project directory.
assert (ROOT / "src").is_dir(), "Open this notebook from the project or notebooks folder."
sys.path.insert(0, str(ROOT / "src"))  # Import the same scientific code used by terminal runs.
def evidence(filename):  # Keep evidence loading consistent across the notebook.
    return json.loads((ROOT / "outputs/results" / filename).read_text(encoding="utf-8"))  # Decode one saved result.
summary = evidence("D_summary.json")  # Read the completed study without recomputing final predictions.
'''


def write(name, title, sections):
    book = nb.v4.new_notebook()  # Start a notebook with a supported file format.
    book.metadata.kernelspec = {"display_name": "Python 3 (sportskpi)", "language": "python", "name": "python3"}  # Use the environment selected at execution.
    book.metadata.authors = [{"name": "Md Forhadul Islam"}]  # Record the study author.
    book.cells = [nb.v4.new_markdown_cell("# " + title + "\n\n" + INTRO), nb.v4.new_markdown_cell("## Find the saved study\n\nA path tells Python where a file lives. JSON stores labelled evidence. A DataFrame is a table with named columns. This setup reads existing results; it cannot change trained weights or the test partition."), nb.v4.new_code_cell(SETUP)]  # Introduce concepts before importing them.
    for prose, code, interpretation in sections:  # Keep purpose, implementation and interpretation adjacent.
        book.cells.extend([nb.v4.new_markdown_cell(prose), nb.v4.new_code_cell(code), nb.v4.new_markdown_cell(interpretation)])  # Explain every analysis cell.
    nb.write(book, ROOT / "notebooks" / name)  # Save an editable notebook for execution and review.


write("01_environment_and_literature.ipynb", "Environment, questions and verified literature", [
("## Why check the hardware?\n\nCUDA lets PyTorch use NVIDIA hardware. Availability alone does not prove that training used it. The saved runtime and fitted booster checks record the actual experiment. A current check below separately describes this session.", '''import torch  # Inspect the installed numerical training library.
runtime = evidence("D_runtime.json")  # Read the hardware used by the completed experiments.
display(pd.Series(runtime, name="Recorded experiment"))  # Keep recorded versions visible for reproduction.
print("CUDA available now:", torch.cuda.is_available())  # Distinguish the current session from historical evidence.
if torch.cuda.is_available():  # Avoid asking a CPU computer for a nonexistent GPU.
    print("Current device:", torch.cuda.get_device_name(0))  # Name the GPU that this session can access.''', "The neural model and XGBoost used CUDA. Logistic regression and random forest used the CPU. GPU use is a computational choice, not evidence that a model predicts better."),
("## What am I asking and why?\n\nA research question must be answerable with the available measurements. Expectations guide the experiment but are not results. The roadmap separates adaptation questions from reliability checks and conditional extensions.", '''display(Markdown((ROOT / "docs/STUDY_ROADMAP.md").read_text(encoding="utf-8")))  # Show the full question sequence beside the computations.''', "I compare models on identical eligible matches, examine confidence as well as ranking, and test whether feature rankings survive changes in seed and training sample. The movement experiment concerns measurement resolution. Hero context is deferred because prospective role timing and inventory completeness are insufficiently established."),
("## How can someone check my literature review?\n\nThe workbook contains sources, search records, short abstract summaries and links between claims and evidence. A short summary is a paraphrase, not a quotation or proof that a whole paper was read. The review states its access limits.", '''sources = evidence("D_literature_sources.json")  # Load the verified source ledger.
print("Source ledger fields:", list(sources))  # Make its structure discoverable without assuming a schema.
display(Markdown((ROOT / "docs/LITERATURE_REVIEW.md").read_text(encoding="utf-8")))  # Display the linked review and its limitations.
print("Workbook:", ROOT / "outputs/literature/Literature_Review_Tracker.xlsx")  # Locate the editable tracker.''', "Credit belongs to the original authors for their methods and findings. My contribution is the independently implemented adaptation, data audit and reliability experiments described here. This is a focused literature review, not an exhaustive systematic review.")])

write("02_data_preparation.ipynb", "From replay records to time observable features", [
("## Why are the clocks and teams important?\n\nA replay clock can keep advancing during a pause. A match clock describes playing time. Confusing them would select the wrong state at a nominal prediction time. A player slot is also not a reliable team label in these files. The pipeline aligns combat event timestamps to preceding states and maps hero identities to the metadata draft.", '''clock = evidence("D_clock_alignment.json")  # Read the audit that motivated event anchored timing.
identity = evidence("D_identity_audit.json")  # Read the evidence behind hero based team assignment.
print("Clock audit fields:", list(clock))  # Expose the audit structure for further inspection.
print("Identity audit fields:", list(identity))  # Keep the team mapping audit discoverable.
display(pd.Series(summary["integrity"], name="Integrity checks"))  # Show checked counts and acknowledged limits.''', "The checks establish internal consistency, not independent ground truth. States strictly precede the selected event anchor. The anchor and state tolerances together give a conservative 2.4 second timing bound. The older max_snapshot_age_seconds field describes only the 1.2 second state matching component; it must not be read as the total bound."),
("## What is a feature and what is the target?\n\nA feature is information available at prediction time. The target is the later outcome we try to predict. Each feature is a currently observed Radiant team total minus the corresponding Dire total. Positive values mean Radiant has more of that statistic. Net worth is economic resources; experience measures progression; a deny prevents an opponent collecting a friendly unit reward.", '''preparation = summary["preparation"]  # Read the exact approved feature list.
display(pd.DataFrame({"feature": preparation["features"]}))  # Show every predictor actually used.
display(pd.DataFrame(preparation["rows_by_horizon_partition"]).pivot(index="horizon_seconds", columns="partition", values="matches"))  # Show how eligibility varies over time.
print("Target:", preparation["target"])  # Name the later result rather than leaving the label implicit.''', "Final duration, final statistics, identifiers and retrospective roles are excluded as predictors. Duration is used only to exclude already completed matches from a live prediction cohort. Towers are available; barracks and trustworthy ward locations are not used. Health and alive counts describe current state, not future survival."),
("## Why split by chronology and whole match?\n\nTraining fits parameters, validation selects checkpoints, calibration adjusts confidence, and the final test estimates later performance after decisions are fixed. Every observation from one match remains in one partition. Whole UTC dates are grouped because exact series identifiers are absent.", '''protocol = summary["protocol"]  # Read the membership fingerprint and frozen settings.
display(pd.Series(protocol["partition_counts"], name="All metadata matches"))  # These counts precede horizon eligibility filtering.
print("UTC boundaries:", protocol["date_boundaries"])  # Show the chronological cut points.
print("Frozen partition SHA256:", protocol["partition_sha256"])  # Identify the exact saved split.
print("Final status:", protocol["status"])  # Make the evaluation boundary explicit.''', "The tables differ because some games are finished or cannot be safely observed at a later horizon. Teams may recur across dates. UTC day grouping is a partial dependence control, not an unseen team or unseen tournament evaluation."),
("## Inspect the prepared table without rebuilding it\n\nThe expensive extraction is implemented in src/dota_data.py. This cell reads its saved table if raw data have been prepared locally. Published notebooks retain their outputs even when a reader has not downloaded the dataset.", '''snapshot_path = ROOT / "data/processed/dota_snapshots.parquet"  # Locate the cached feature table.
if snapshot_path.exists():  # Permit a lightweight read on the original study computer.
    snapshots = pd.read_parquet(snapshot_path)  # Load already extracted observations.
    assert snapshots.groupby("match_id").partition.nunique().max() == 1  # Check that repeated times never cross partitions.
    display(snapshots.drop(columns=["radiant_win"], errors="ignore").head())  # Inspect inputs without using outcomes to choose a method.
else:  # A clean clone intentionally excludes raw and processed data.
    print("Run the documented download and preparation commands to inspect local rows.")  # Explain the missing cache honestly.''', "Preparation can be reproduced using the commands in the README. I keep the original files separate from derived tables so that a transformation never silently changes source data.")])

write("03_model_comparison.ipynb", "Neural adaptation and baseline comparisons", [
("## How do the four models differ?\n\nLogistic regression learns a weighted sum of features and converts it into a probability. A random forest averages decision trees. XGBoost adds trees that correct earlier errors. The neural network learns two layers of nonlinear combinations. Scaling is fitted on training data for logistic regression and the neural network. Validation controls the neural checkpoint; it never uses final test outcomes.", '''display(pd.Series(summary["protocol"]["model_settings"], name="Frozen settings"))  # Show actual hyperparameters, not retrospectively selected ones.
print("Seeds:", summary["protocol"]["seeds"])  # Seeds make stochastic training repeatable.
print("Source implementation:", ROOT / "src/dota_models.py")  # Direct readers to the fully commented shared model code.''', "The neural architecture follows the original paper's two hidden layers, with adaptations for binary team differences and time observable Dota inputs. These are fixed reference settings, not an exhaustive hyperparameter search. Repeating deterministic logistic regression produces effectively zero seed spread."),
("## Where are the actual fitting results?\n\nThe saved development records come from fitting on training matches and measuring validation predictions. Each fitted model was cached. Training can be deliberately repeated with the development command in the README; this teaching notebook does not overwrite the frozen experiment.", '''development = pd.DataFrame(evidence("D_development_metrics.json")["records"])  # Read actual per seed fits and validation measurements.
display(development.loc[(development.horizon_seconds == 1200) & (development.probability == "raw"), ["model", "seed", "matches", "auc", "training_seconds"]])  # Keep the training evidence separate from final results.
display(Image(filename=str(ROOT / "outputs/figures/D_learning_curve.png")))  # Display the saved neural development loss curve.''', "An epoch is one pass through the training data. Early stopping restores the best validation checkpoint rather than the last epoch. The curve describes a particular fitted run; it does not establish that the neural model is universally optimal."),
("## What do the metrics mean?\n\nROC AUC measures whether winners receive higher scores than losers across possible thresholds. Accuracy uses a probability threshold of 0.5. Brier score is the mean squared error of predicted probabilities. Log loss penalises confident mistakes. Higher AUC and accuracy are better; lower Brier score and log loss are better. Seed standard deviation describes training variation, not a confidence interval over new datasets.", '''test = pd.DataFrame(summary["test_table"])  # Use the completed, saved final evaluation.
primary = test.loc[(test.horizon_seconds == 1200) & (test.probability == "raw")]  # Display the prespecified primary time and raw probabilities.
display(primary[["model", "matches", "auc_mean", "auc_sd", "accuracy_mean", "brier_mean", "log_loss_mean"]].round(6))  # Show all four models on the same eligible matches.
display(Image(filename=str(ROOT / "outputs/figures/D_performance.png")))  # Compare the three time horizons without refitting.''', "Logistic regression has a small advantage over the adapted neural model at the primary horizon in this dataset. This does not refute the original LoL findings or establish the best possible Dota model. Later time points have different eligible match populations."),
("## Is the difference larger than match sample uncertainty?\n\nA paired bootstrap samples whole matches with replacement and measures both models on the same sampled games. The day block version samples whole UTC days, keeping same day games together. These are conditional intervals for the fitted predictions; they do not represent every source of uncertainty.", '''display(pd.DataFrame(summary["primary_paired_comparison"]["match"]))  # Show lower and upper interval endpoints for logistic minus neural.
display(pd.DataFrame(summary["primary_paired_comparison"]["utc_day_block"]))  # Check sensitivity to same day dependence.
display(pd.Series(summary["primary_paired_comparison"]["observed_logistic_minus_neural"]))  # Distinguish seed averaged predictions from the mean of seed metrics.
display(pd.DataFrame(summary["common_cohort_neural"]))  # Hold match membership fixed when comparing neural prediction times.''', "The AUC intervals favour logistic regression for this fixed comparison. The accuracy interval includes zero. AUC of averaged predictions need not equal the average of per seed AUC values. The common horizon cohort controls one selection difference, but describes games with observations at every horizon rather than all games.")])

write("04_calibration_and_kpi_stability.ipynb", "Probability calibration and stability of KPI explanations", [
("## Can a correct ranking still have poor confidence?\n\nYes. Ranking and confidence are different properties. Temperature scaling divides the model's log odds by one positive fitted number. A distinct calibration period fits that number. Both raw and adjusted probabilities were prespecified for reporting, so an unhelpful adjustment is still shown.", '''table = pd.DataFrame(summary["test_table"])  # Use the same final predictions for both versions.
display(table.loc[table.horizon_seconds == 1200, ["model", "probability", "brier_mean", "brier_sd", "log_loss_mean", "log_loss_sd"]].round(6))  # Report probability losses, including unsuccessful adjustments.
display(Image(filename=str(ROOT / "outputs/figures/D_calibration.png")))  # Display observed win frequencies against predicted confidence.''', "A reliability diagram groups predictions into probability bins. Sparse bins can be noisy, so the bin counts are saved in D_calibration_bins.json. Temperature scaling is not guaranteed to improve an unseen period, especially when the calibration sample is small or the distribution changes."),
("## What does SHAP explain?\n\nSHAP attributes a model prediction relative to a reference distribution of training examples. Mean absolute SHAP summarises predictive influence, not causal impact. Exact linear and tree explainers and a deep explainer are used for the corresponding models. Their output units differ, so I compare ranks across model families rather than treating magnitudes as equivalent.", '''explanations = evidence("D_explanations.json")  # Read the actual explanation experiment and assumptions.
display(pd.Series(explanations["output_units"], name="Explanation units"))  # Prevent invalid magnitude comparisons across models.
display(pd.DataFrame(summary["development_rank_summary"]).query("horizon_seconds == 1200"))  # Show the primary development rankings across seed runs.
display(Image(filename=str(ROOT / "outputs/figures/D_rank_stability.png")))  # Compare seed ranks and actual training resample stability.''', "Net worth is consistently prominent. Experience, combat and survival measures can share predictive information. A correlated feature can change rank when another feature or model absorbs its signal. A low SHAP rank is not evidence that an action is strategically unimportant."),
("## Why use three kinds of uncertainty?\n\nChanging the seed measures training randomness. Resampling explained matches with fixed weights measures explanation sample variation. Refitting on resampled training matches measures sensitivity to which training examples were observed. These procedures answer different questions and must not be labelled interchangeably.", '''display(pd.DataFrame(summary["seed_rank_agreement"]))  # Compare feature rank ordering between seed runs.
refits = pd.DataFrame(summary["training_refit_rank_summary"])  # Read thirty real refits per model, not resampled fixed SHAP values.
display(refits.pivot(index="feature", columns="model", values="top3_frequency").round(3))  # Show how often each feature enters the top three.
print("Refit scope:", evidence("D_refit_stability.json")["scope"])  # State the conditional design and its limits.''', "The refits keep the estimator seed fixed and reuse a fixed validation explanation set to isolate training sample sensitivity. Thirty refits provide a descriptive stability check, not precise tail probability estimates. KPI stability remains conditional on these models, feature definitions and professional matches.")])

write("05_movement_sampling.ipynb", "What minute sampling does to a movement measure", [
("## Why revisit movement resolution?\n\nThe original paper identifies position sampling as a limitation. I study a related measurement issue in Dota 2 using development matches only. Summing distances between approximately one second positions gives an operational sampled path. Joining only minute endpoints omits bends. Neither measurement is continuous ground truth or a direct measure of tactical quality.", '''movement = summary["movement"]  # Read the completed development measurement experiment.
display(pd.DataFrame(movement["summaries"]))  # Show one match level summary per sampling offset.
print("Native sampling:", movement["native_sampling"])  # State the event anchored timing approximation.
print("Interpretation:", movement["interpretation"])  # Keep measurement scope beside the results.''', "The retained fraction is coarse endpoint displacement divided by the native sampled path over the same segment endpoints. The triangle inequality already implies that endpoint chords cannot exceed the corresponding path. The empirical question is the conditional magnitude and its sensitivity to sampling offset."),
("## Why exclude deaths, gaps and implausible jumps?\n\nA line drawn across a teleport or death can be a fictitious walking distance. The implementation splits at those boundaries, rejects timing gaps and uses a documented speed rule. It never compares unmatched native and coarse endpoints. These rules also remove genuine rapid movement and reduce coverage.", '''display(pd.Series({key: movement[key] for key in ["window_seconds", "coarse_interval_seconds", "offsets_seconds", "speed_multiplier", "jump_slack_coordinate_units", "max_gap_seconds", "unique_development_matches"]}))  # Show the actual operational rules.
display(Image(filename=str(ROOT / "outputs/figures/D_movement.png")))  # Show distributions rather than only a pooled mean.
display(pd.Series(movement["common_offset_medians"], name="Median retained fraction on common matches"))  # Compare grid offsets while holding matches fixed.
print("Matches supporting every offset:", movement["common_offset_matches"])  # Disclose the restricted common cohort.''', "Only a subset supplies continuous matched minute segments. Results describe that subset and cannot be extrapolated to every player minute or every match. Coordinate units are not claimed to be metres. Offset comparisons are correlated observations, not independent replications."),
("## Does this prove movement changes winning KPIs?\n\nNo. This experiment measures sampling distortion. Movement was not added to the prediction features, so it does not establish a change in predictive accuracy or KPI ranking. Such a claim would need a new prospectively designed experiment and an untouched evaluation set.", '''for limitation in movement["limits"]:  # Present all saved qualifications without silently dropping inconvenient ones.
    print(limitation)  # Keep the implementation's own warnings visible to a beginner.''', "The result supports careful measurement design. It does not reproduce the magnitude of the LoL paper's movement results, identify causal winning behaviour or resolve the original authors' full spatial analysis.")])

write("06_findings_and_next_steps.ipynb", "What I learned, what remains unanswered and how to rerun", [
("## Put the questions next to the evidence\n\nA useful conclusion states what was observed, which question it answers and where it stops. The following table separates the completed adaptation and reliability checks from extensions that the data do not yet support.", '''findings = [  # Keep the interpretation linked to measured evidence rather than an assumed positive outcome.
    ("Neural adaptation", "Runs on time observable Dota team features", "Different game and inputs; no numerical LoL reproduction"),
    ("Model comparison", "Logistic regression has a small primary AUC advantage", "Fixed settings; no universal superiority"),
    ("Calibration", "Raw and temperature losses are both reported", "Adjustment is not uniformly beneficial"),
    ("KPI stability", "Net worth is prominent across models and resamples", "Predictive association; correlated features redistribute credit"),
    ("Movement resolution", "Minute chords retain less sampled path length", "Restricted continuous segments; no downstream prediction test"),
    ("Hero and item context", "Deferred following the audit", "Prospective roles and inventory completeness unresolved")]
display(pd.DataFrame(findings, columns=["Question", "Observed answer", "Limit"]))  # Make the scope easy to inspect.''', "This independent study adds a transparent audit and several reliability checks to an adapted pipeline. It does not claim a novel prediction algorithm or credit for the original research method."),
("## Reproduce the computation in the right order\n\nThe README documents environment setup, pinned downloads, clock preparation, model fitting, explanations, movement analysis, protocol freeze and the single final evaluation. The final command is guarded against accidental repetition. These notebooks read the completed evidence so that rerunning a teaching cell never silently changes the final experiment.", '''print("Shared implementation:", ROOT / "src")  # Locate reusable and commented scientific code.
print("Full run instructions:", ROOT / "README.md")  # Keep the operational sequence in one maintained location.
print("Review tracker:", ROOT / "outputs/literature/Literature_Review_Tracker.xlsx")  # Locate verified source records.
print("Frozen test status:", summary["protocol"]["status"])  # Remind readers that the test has already been used.
print("Saved limitations:")  # Introduce the recorded qualifications.
for limitation in summary["limits"]:  # Preserve every result ledger limitation.
    print(limitation)  # Display one plain language statement at a time.''', "Future experiments should use a new held out period. Useful next steps include better prospective role and inventory provenance, team or series identifiers, and movement features defined before seeing new outcomes. Rugby transfer would require a separately licensed and audited rugby dataset.")])
print("Created six teaching notebooks.")  # Provide a concise completion signal.

# Put the implementation beside its explanations without executing a new fit.
implementations = {  # Choose the shared functions that answer each notebook's main implementation question.
    "02_data_preparation.ipynb": ("dota_data", ["latest_rows"]),
    "03_model_comparison.ipynb": ("dota_models", ["NeuralPredictor", "fit_model"]),
    "04_calibration_and_kpi_stability.ipynb": ("dota_explanations", ["explain", "ranking_summary"]),
    "05_movement_sampling.ipynb": ("dota_movement", ["trajectory_pairs"])}
for filename, (module, functions) in implementations.items():  # Add the exact live source to its teaching notebook.
    path = ROOT / "notebooks" / filename  # Select the notebook just written above.
    book = nb.read(path, as_version=4)  # Preserve its question, code and interpretation cells.
    book.cells.append(nb.v4.new_markdown_cell("## Read the shared implementation line by line\n\nThe source below is the actual function used by the pipeline, including comments explaining what each operation does and why. `inspect` reads Python source; it does not call the function or refit a model. A function is a named sequence of reusable steps; a class groups a model's state and prediction methods. Read the comments alongside the preceding explanation, then open the source file to explore further. The frozen result record retains the original implementation fingerprints; later documentation changes do not replace that historical record."))  # Explain why source is displayed rather than rerunning the final experiment.
    source = "import inspect  # Read implementation text without running a scientific experiment.\nfrom IPython.display import Code  # Render Python source with readable syntax highlighting.\nimport " + module + "  # Access the same shared module used by the terminal pipeline.\n"  # Keep the display code itself beginner readable.
    source += "\n".join("display(Code(inspect.getsource(" + module + "." + function + "), language='python'))  # Show the commented implementation without calling it." for function in functions)  # Include every selected function.
    book.cells.append(nb.v4.new_code_cell(source))  # Save executable source display cells with their explanations.
    nb.write(book, path)  # Keep the notebook editable as a normal Jupyter document.
