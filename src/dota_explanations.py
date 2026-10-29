"""Explain fitted models and quantify two distinct kinds of ranking uncertainty.

SHAP explains predictions, not causation. Comparing ranks avoids comparing
incompatible absolute attribution units across probability and log odds outputs.
"""
import os  # Set deterministic CUDA configuration before loading model helpers.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Make GPU refits reproducible.
import itertools  # Enumerate pairs of models and seeds without manual lists.
import json  # Read resumable result checkpoints.
import time  # Report the actual computation time.

import joblib  # Load the fitted development models unchanged.
import numpy as np  # Calculate bootstrap samples and feature summaries.
import pandas as pd  # Read prepared match tables and organise rankings.
from scipy.stats import rankdata, spearmanr  # Handle tied ranks and measure ranking agreement.
import shap  # Use documented tree and neural explanation implementations.
import torch  # Place neural explanation tensors on the GPU.
from dota_data import PROCESSED, ROOT, save_json  # Reuse project paths and result saving.
from dota_models import fit_model, metrics  # Reuse exactly the same estimator settings for refits.


def explain(model, name, background, observations):
    """Explain the positive class with a training only reference background.

    Tree and neural explainers differ in approximation details. Their ranks
    are descriptive comparisons; their absolute numerical scales differ.
    """
    if name == "logistic":  # A linear log odds function has an analytic interventional attribution.
        scaler, linear = model[0], model[1]  # Separate the fitted transform from its coefficients.
        values = (scaler.transform(observations) - scaler.transform(background).mean(0)) * linear.coef_[0]  # Attribute each scaled feature's deviation from the training reference.
    elif name == "neural_network":  # DeepExplainer attributes the network's single output logit.
        model.network.eval()  # Disable dropout so explanations refer to a fixed prediction function.
        reference = torch.tensor(model.scaler.transform(background), dtype=torch.float32, device="cuda")  # Reuse training scaling and reference rows.
        points = torch.tensor(model.scaler.transform(observations), dtype=torch.float32, device="cuda")  # Explain only designated development observations.
        values = shap.DeepExplainer(model.network, reference).shap_values(points)  # Check additivity using SHAP's default check.
        values = np.asarray(values)[..., 0]  # Remove the singleton output axis, retaining rows and features.
    else:  # Tree models use the same observed training background rows.
        values = shap.TreeExplainer(model, data=background, feature_perturbation="interventional").shap_values(observations)  # State the interventional assumption explicitly.
        values = np.asarray(values)  # Convert the explainer output into a predictable array.
        if values.ndim == 3:  # Random Forest produces one attribution array per class.
            values = values[..., 1]  # Select Radiant win, not the opposite class.
    assert values.shape == observations.shape  # Prevent accidentally ranking outputs instead of features.
    assert np.isfinite(values).all()  # Reject invalid explanations rather than silently dropping them.
    return values  # Return one attribution for each observation and feature.


def ranking_summary(values, features, draws=1000, seed=2718):
    """Resample explained matches while holding model weights fixed."""
    magnitude = np.abs(values)  # Global importance uses mean absolute contribution, not a causal effect.
    mean = magnitude.mean(0)  # Average across the explained development matches.
    rng = np.random.default_rng(seed)  # Fix the explanation sampling bootstrap independently of training.
    ranks = []  # Collect rank vectors from resampled observed matches.
    for _ in range(draws):  # Repeat the match sampling procedure.
        index = rng.integers(0, len(values), len(values))  # Resample whole match explanation rows.
        ranks.append(rankdata(-magnitude[index].mean(0)))  # Rank larger importance first and average ties.
    ranks = np.asarray(ranks)  # Shape is bootstrap samples by features.
    return [{"feature": feature, "mean_absolute_shap": float(mean[i]), "rank": float(rankdata(-mean)[i]), "rank_95_interval": np.quantile(ranks[:, i], [.025, .975]).tolist(), "top3_frequency": float((ranks[:, i] <= 3).mean())} for i, feature in enumerate(features)]  # Keep importance, uncertainty and inclusion frequency together.


def run():
    """Produce development explanations and exploratory training sample refits."""
    frame = pd.read_parquet(PROCESSED / "dota_snapshots.parquet")  # Read the same prepared feature definitions as training.
    features = [c for c in frame if c.endswith("_diff")]  # Exclude outcomes, dates and identities.
    records, arrays, seed_vectors = [], {}, {}  # Store interpretable summaries and underlying attribution evidence.
    for horizon in [600, 1200, 1800]:  # Explain each prespecified time separately.
        subset = frame.loc[frame.horizon_seconds == horizon]  # Do not combine different game states into one ranking.
        background = subset.loc[subset.partition == "train", features].sample(64, random_state=314)  # Use identical training reference rows across models.
        observations = subset.loc[subset.partition == "validation"].sample(256, random_state=314)  # Use identical held out development rows across models.
        for name in ["logistic", "random_forest", "xgboost", "neural_network"]:  # Preserve the agreed model comparison.
            for seed in [0, 1, 2]:  # Inspect explanation variation caused by the fixed training seeds.
                model = joblib.load(ROOT / f"outputs/models/{name}_{horizon}_{seed}.joblib")["model"]  # Load the original fitted model without retraining it.
                values = explain(model, name, background, observations[features])  # Generate one model's explanations for the common observations.
                key = f"{name}_{horizon}_{seed}"  # Use the experiment identity as an array key.
                arrays[key] = values  # Save underlying contributions for reproducible summaries.
                seed_vectors[key] = rankdata(-np.abs(values).mean(0))  # Keep ranks for cross seed and model agreement.
                records.append({"horizon_seconds": horizon, "model": name, "seed": seed, "explained_matches": len(observations), "background_matches": len(background), "ranking": ranking_summary(values, features)})  # Save bootstrap uncertainty for the fixed fitted model.
                print(f"Explained {horizon // 60} min {name} seed {seed}", flush=True)  # Report progress without printing hundreds of attribution rows.
    pairwise = []  # Compare ranks rather than incompatible raw output units.
    for horizon in [600, 1200, 1800]:  # Keep agreement comparisons within one observation time.
        names = ["logistic", "random_forest", "xgboost", "neural_network"]  # Define the model labels used in saved keys.
        means = {name: np.mean([seed_vectors[f"{name}_{horizon}_{seed}"] for seed in [0, 1, 2]], axis=0) for name in names}  # Summarise the seed specific rank vectors.
        for left, right in itertools.combinations(names, 2):  # Enumerate each model pair once.
            pairwise.append({"horizon_seconds": horizon, "left": left, "right": right, "spearman_rank_agreement": float(spearmanr(means[left], means[right]).statistic)})  # Measure monotonic ordering agreement, not causal agreement.
    np.savez_compressed(ROOT / "outputs/results/D_shap_values.npz", **arrays)  # Preserve compact numerical explanation evidence.
    save_json("D_explanations.json", {"shap_version": shap.__version__, "features": features, "records": records, "model_rank_agreement": pairwise, "bootstrap_draws": 1000, "output_units": {"logistic": "log odds", "xgboost": "log odds", "neural_network": "logits", "random_forest": "probability"}, "assumptions": ["Interventional reference background; correlated features require caution", "DeepExplainer approximates nonlinear neural attributions", "Absolute magnitudes are not directly compared across model output units", "Fixed model bootstrap intervals measure explained match sampling uncertainty, not refitting uncertainty"]})  # Record the explanation assumptions beside the results.
    refit_path = ROOT / "outputs/results/D_refit_stability.json"  # Checkpoint the more expensive training sample experiment separately.
    previous = json.loads(refit_path.read_text()) if refit_path.exists() else {"records": []}  # Resume completed bootstrap refits if a run was interrupted.
    completed = {(r["model"], r["replicate"]) for r in previous["records"]}  # Identify fits that should not be repeated.
    subset = frame.loc[frame.horizon_seconds == 1200]  # Prespecify the middle observation time for exploratory refits.
    train = subset.loc[subset.partition == "train"]  # Resample training matches only.
    validation = subset.loc[subset.partition == "validation"]  # Keep the development checkpoint and evaluation period fixed.
    background = train[features].sample(64, random_state=314)  # Hold the explanation background fixed to isolate refitting differences.
    observations = validation.sample(128, random_state=314)  # Bound the expensive explanation work while retaining identical rows.
    for name in ["logistic", "random_forest", "xgboost", "neural_network"]:  # Refit every agreed model family.
        for replicate in range(30):  # Use exploratory bootstrap replicates rather than claiming precise tail inference.
            if (name, replicate) in completed:  # Do not repeat a completed stochastic fit during resume.
                continue  # Move directly to the next missing replicate.
            rng = np.random.default_rng(10000 + replicate)  # Give every model the same bootstrap sample for a given replicate.
            sample = train.iloc[rng.integers(0, len(train), len(train))]  # Draw complete training match rows with replacement.
            model, seconds = fit_model(name, sample[features], sample.radiant_win.astype(int), validation[features], validation.radiant_win.astype(int), 0)  # Fix the estimator seed to focus on changed training samples.
            p = model.predict_proba(validation[features])[:, 1]  # Assess raw predictions on development matches, never final test.
            values = explain(model, name, background, observations[features])  # Explain the same small development panel for each fitted sample.
            importance = np.abs(values).mean(0)  # Average attribution magnitudes for a global ranking.
            previous["records"].append({"model": name, "replicate": replicate, "training_seconds": seconds, "metrics": metrics(validation.radiant_win.astype(int).to_numpy(), p), "importance": importance.tolist(), "ranks": rankdata(-importance).tolist()})  # Preserve actual refit observations rather than simulated rank noise.
            previous.update(horizon_seconds=1200, features=features, replicates_per_model=30, explained_matches=128, background_matches=64, scope="Exploratory training match bootstrap; development only; fixed estimator seed and reference background")  # Record the experiment's computational and inferential limits.
            save_json("D_refit_stability.json", previous)  # Checkpoint after each completed fit.
            print(f"Refit stability: {name} {replicate + 1}/30", flush=True)  # Make the GPU job easy to monitor at intervals.
    return records  # Return the primary explanations for interactive notebook use.


if __name__ == "__main__":  # Command line use shares the notebook implementation.
    run()  # Produce explanations and both uncertainty assessments.
