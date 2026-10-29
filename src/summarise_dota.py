"""Summarise frozen predictions and development explanations without refitting."""
import json  # Read measured result files rather than manually typing numbers.
import numpy as np  # Calculate averages, spread and paired bootstrap intervals.
import pandas as pd  # Join match identities and organise report tables.
from scipy.stats import spearmanr  # Compare ordered feature rankings across seeds.
from dota_data import PROCESSED, ROOT, save_json  # Share the result folder and evidence writer.
from dota_models import metrics  # Reuse the same metric definitions as final evaluation.


def read(name):
    return json.loads((ROOT / "outputs/results" / name).read_text(encoding="utf-8"))  # Load one saved evidence file from its stable location.


def run():
    final = read("D_final_metrics.json")  # Final predictions already exist; this script never evaluates models again.
    frame = pd.DataFrame(final["records"])  # Organise per seed measurements into rows.
    table = []  # Build a machine readable table shared by README and report.
    for (horizon, model, probability), group in frame.groupby(["horizon_seconds", "model", "probability"]):  # Keep experimental conditions distinct.
        record = dict(horizon_seconds=int(horizon), model=model, probability=probability, matches=int(group.matches.iloc[0]))  # Attach the exact eligible test population size.
        for measurement in ["auc", "accuracy", "brier", "log_loss"]:  # Report both ranking and probability performance.
            record[measurement + "_mean"] = float(group[measurement].mean())  # Average separately fitted seed results.
            record[measurement + "_sd"] = float(group[measurement].std(ddof=1))  # Seed spread is not a confidence interval over independent datasets.
        table.append(record)  # Preserve every model and both probability versions.
    predictions = pd.read_parquet(ROOT / "outputs/predictions/D_final_predictions.parquet")  # Work only with frozen predictions from the completed final stage.
    average = predictions.groupby(["match_id", "horizon_seconds", "model"], as_index=False).agg(y=("y", "first"), raw=("raw", "mean"), temperature=("temperature", "mean"))  # Define the prespecified seed averaged probabilities for paired comparisons and plots.
    primary = average.loc[average.horizon_seconds == 1200]  # Use the middle observation time chosen before final evaluation.
    left = primary.loc[primary.model == "logistic"].set_index("match_id").sort_index()  # Align the linear reference by actual match identity.
    right = primary.loc[primary.model == "neural_network"].set_index("match_id").sort_index()  # Align the neural reference to the same matches.
    assert left.index.equals(right.index) and np.array_equal(left.y, right.y)  # Paired uncertainty requires identical observations and outcomes.
    y, pl, pr = left.y.to_numpy(), left.raw.to_numpy(), right.raw.to_numpy()  # Extract arrays without changing probabilities.
    snapshots = pd.read_parquet(PROCESSED / "dota_snapshots.parquet")  # Read dates for the prespecified cluster sensitivity check.
    dates = snapshots.drop_duplicates("match_id").set_index("match_id").loc[left.index, "start_time"]  # Look up the same matches' recorded dates.
    days = pd.to_datetime(dates, unit="s", utc=True).dt.floor("D").astype(str).to_numpy()  # Group games played on one UTC day.
    groups = [np.flatnonzero(days == day) for day in np.unique(days)]  # Retain complete day blocks when resampling.
    paired = {}  # Keep match and day block uncertainty separate.
    for scheme in ["match", "utc_day_block"]:  # Assess sensitivity to dependence between games on a day.
        rng = np.random.default_rng(811)  # Use a fixed resampling seed for transparent reproduction.
        differences = []  # Store paired metric differences for one scheme.
        for _ in range(1000):  # Produce prespecified descriptive percentile intervals.
            index = rng.integers(0, len(y), len(y)) if scheme == "match" else np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])  # Sample matches or whole days with replacement.
            if len(np.unique(y[index])) < 2:  # AUC needs both classes in a bootstrap resample.
                continue  # Skip the exceptional undefined sample.
            a, b = metrics(y[index], pl[index]), metrics(y[index], pr[index])  # Compare the same resampled matches for both models.
            differences.append({key: a[key] - b[key] for key in a})  # Use a consistent linear minus neural direction.
        paired[scheme] = {key: np.quantile([row[key] for row in differences], [.025, .975]).tolist() for key in differences[0]}  # Preserve the uncertainty bounds for all metrics.
    observed_left, observed_right = metrics(y, pl), metrics(y, pr)  # Calculate the actual seed averaged comparison.
    paired["observed_logistic_minus_neural"] = {key: observed_left[key] - observed_right[key] for key in observed_left}  # State the measured effect rather than only a significance label.
    paired["matches"], paired["utc_days"] = len(y), len(groups)  # Record the sample and cluster counts.
    common_sets = [set(average.loc[(average.model == "neural_network") & (average.horizon_seconds == t), "match_id"]) for t in [600, 1200, 1800]]  # Identify identical final matches across observation times.
    common_ids = set.intersection(*common_sets)  # Avoid mistaking a changed survivor cohort for a pure time effect.
    common = []  # Save the reference network's matched cohort measurements.
    for horizon in [600, 1200, 1800]:  # Keep the same matches at every reported time.
        subset = average.loc[(average.model == "neural_network") & (average.horizon_seconds == horizon) & average.match_id.isin(common_ids)]  # Select frozen probabilities only.
        common.append({"horizon_seconds": horizon, "matches": len(subset), **metrics(subset.y.to_numpy(), subset.raw.to_numpy())})  # Report this as a conditional long match cohort.
    explanations = read("D_explanations.json")  # Explanations were generated on development matches before test access.
    ranks = []  # Organise feature level development evidence.
    for record in explanations["records"]:  # Read every model and seed rather than selecting one attractive plot.
        ranks.extend({"model": record["model"], "seed": record["seed"], "horizon_seconds": record["horizon_seconds"], **row} for row in record["ranking"])  # Preserve each feature's interpretation scope.
    rank_frame = pd.DataFrame(ranks)  # Prepare summaries and heatmaps from the saved rank table.
    rank_summary = rank_frame.groupby(["horizon_seconds", "model", "feature"], as_index=False).agg(mean_rank=("rank", "mean"), mean_absolute_shap=("mean_absolute_shap", "mean"), fixed_model_top3_frequency=("top3_frequency", "mean")).to_dict("records")  # Compare ranks, while retaining original attribution units separately.
    seed_agreement = []  # Quantify within model changes from training randomness.
    for (horizon, model), group in rank_frame.groupby(["horizon_seconds", "model"]):  # Compare only like observation times and model families.
        pivot = group.pivot(index="feature", columns="seed", values="rank")  # Align feature orders before calculating rank correlation.
        values = [float(spearmanr(pivot[a], pivot[b]).statistic) for a, b in [(0, 1), (0, 2), (1, 2)]]  # Use all three seed pairs.
        seed_agreement.append(dict(horizon_seconds=int(horizon), model=model, pairwise_spearman=values, mean_spearman=float(np.mean(values))))  # Save descriptive agreement rather than claiming independent inferential samples.
    refits = read("D_refit_stability.json")  # Training sample bootstrap is distinct from resampling a fixed model's explanations.
    refit_summary = []  # Summarise exploratory refitting uncertainty for each feature.
    for model in ["logistic", "random_forest", "xgboost", "neural_network"]:  # Include every model family even when its rankings are unstable.
        rows = [row for row in refits["records"] if row["model"] == model]  # Collect this model's thirty actual bootstrap fits.
        array = np.asarray([row["ranks"] for row in rows])  # Shape is training resamples by features.
        for i, feature in enumerate(refits["features"]):  # Preserve the common approved feature vocabulary.
            refit_summary.append(dict(model=model, feature=feature, refits=len(rows), mean_rank=float(array[:, i].mean()), rank_iqr=np.quantile(array[:, i], [.25, .75]).tolist(), top3_frequency=float((array[:, i] <= 3).mean())))  # Report inclusion frequency and spread without overstating precise tail intervals.
    movement = read("D_movement.json")  # The measurement experiment used development trajectories only.
    movement_pairs = pd.read_parquet(ROOT / "outputs/results/D_movement_match_pairs.parquet")  # Use the saved matched endpoint comparisons.
    offset_sets = [set(movement_pairs.loc[movement_pairs.offset_seconds == offset, "match_id"]) for offset in [0, 15, 30, 45]]  # Find which matches support each sampling grid.
    common_movement_ids = set.intersection(*offset_sets)  # Hold the population fixed when discussing grid phase differences.
    movement["common_offset_matches"] = len(common_movement_ids)  # Disclose the smaller population supporting every offset.
    movement["common_offset_medians"] = {str(offset): float(movement_pairs.loc[(movement_pairs.offset_seconds == offset) & movement_pairs.match_id.isin(common_movement_ids), "retained_fraction"].median()) for offset in [0, 15, 30, 45]}  # Do not compare offset medians while silently changing cohorts.
    save_json("D_summary.json", {"test_table": table, "primary_paired_comparison": paired, "common_cohort_neural": common, "development_rank_summary": rank_summary, "seed_rank_agreement": seed_agreement, "training_refit_rank_summary": refit_summary, "movement": movement, "runtime": read("D_runtime.json"), "preparation": read("D_preparation.json"), "integrity": read("D_integrity_validation.json"), "protocol": read("D_protocol.json"), "limits": ["Fixed model bootstrap, training seed spread and bootstrap refitting describe different uncertainties", "UTC day blocks are a dependence sensitivity check, not verified tournament series", "No causal or novelty claim", "Hero context extension deferred for provenance and completeness reasons"]})  # Supply one generated source for every report and README number.
    average.to_parquet(ROOT / "outputs/results/D_seed_averaged_predictions.parquet", index=False)  # Preserve the exact arrays underlying curves and paired comparisons.
    print(frame.loc[(frame.horizon_seconds == 1200) & (frame.probability == "raw")].groupby("model")[["auc", "accuracy", "brier", "log_loss"]].mean().round(4).to_string())  # Show the main result without hiding all seed records.
    print("Primary paired AUC interval:", paired["match"]["auc"], "day block:", paired["utc_day_block"]["auc"], flush=True)  # Keep effect uncertainty visible alongside headline scores.
    return table  # Let the summary notebook display the generated table.


if __name__ == "__main__":  # Importing plotting helpers does not recompute results automatically.
    run()  # Summarise the saved final predictions deliberately.
