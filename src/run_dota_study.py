"""Run the prespecified model comparison while keeping the final test locked.

Use --stage development first. The final stage requires a frozen protocol and
refuses to evaluate twice. Saved predictions support later figures and intervals.
"""
import argparse  # Choose development or final evaluation explicitly.
import json  # Read and update the recorded experimental protocol.
import os  # Configure deterministic CUDA behaviour before importing PyTorch.
import platform  # Record the operating system for reproducibility.
import time  # Measure the duration of complete runs.
from pathlib import Path  # Handle project paths consistently.

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Required by deterministic CUDA matrix multiplication.
import joblib  # Save fitted models so final evaluation does not retrain them.
import numpy as np  # Manipulate predictions and random bootstrap samples.
import pandas as pd  # Organise experiment records and prediction tables.
import sklearn  # Record the exact installed machine learning version.
import torch  # Verify the requested GPU and report its memory use.
import xgboost  # Record the boosted tree implementation version.
from dota_data import PROCESSED, ROOT, save_json  # Reuse project paths and evidence saving.
from dota_models import apply_temperature, fit_model, metrics, temperature_fit  # Share model and evaluation definitions with notebooks.
from common import SEEDS  # Keep training seeds in the one shared project definition.

MODELS = ["logistic", "random_forest", "xgboost", "neural_network"]  # Compare the agreed model families.
MODEL_DIR = ROOT / "outputs/models"  # Keep potentially large fitted objects local.
PRED_DIR = ROOT / "outputs/predictions"  # Save compact predictions for audit and plotting.


def bootstrap_interval(y, p, seed=1729, count=1000):
    """Resample matches for a fixed prediction vector, not for model refitting."""
    rng = np.random.default_rng(seed)  # Use a fixed bootstrap generator separate from model training.
    values = []  # Store measurements from resampled matches.
    for _ in range(count):  # Repeat enough times for descriptive percentile intervals.
        index = rng.integers(0, len(y), len(y))  # Sample whole match observations with replacement.
        if len(np.unique(y[index])) < 2:  # AUC is undefined if a resample has only one outcome class.
            continue  # Ignore that exceptional resample rather than inventing an AUC.
        values.append(metrics(y[index], p[index]))  # Evaluate the same frozen predictions on the resample.
    return {name: np.quantile([v[name] for v in values], [.025, .975]).tolist() for name in values[0]}  # Return percentile bounds for each metric.


def run(stage="development"):
    """Fit on training data or evaluate frozen models on the final period."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Create the local fitted model folder.
    PRED_DIR.mkdir(parents=True, exist_ok=True)  # Create the saved prediction folder.
    protocol_path = ROOT / "outputs/results/D_protocol.json"  # Locate the pretraining split and settings record.
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))  # Read the frozen membership description.
    final_path = ROOT / "outputs/results/D_final_metrics.json"  # A final result file also acts as a rerun guard.
    if stage == "final":  # Require a deliberate transition before reading final outcomes for evaluation.
        if final_path.exists():  # Final results should be read, not regenerated to select a better run.
            raise RuntimeError("Final evaluation already exists. Read saved predictions and results instead.")  # Prevent repeated test evaluation.
        assert protocol.get("analysis_frozen") is True  # A development run cannot accidentally open the final test.
    assert torch.cuda.is_available(), "The requested GPU must be available."  # Refuse silent CPU neural training.
    gpu_name = torch.cuda.get_device_name(0)  # Verify the physical device used for the run.
    probe = torch.ones((256, 256), device="cuda")  # Allocate a small tensor on the GPU.
    assert float((probe @ probe)[0, 0]) == 256  # Check that an actual CUDA computation succeeds.
    torch.cuda.synchronize()  # Complete the test before recording hardware details.
    save_json("D_runtime.json", {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda, "numpy": np.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "gpu": gpu_name, "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory, "cuda_probe_passed": True, "neural_device": "cuda", "cpu_threads_for_classical_models": 4})  # Preserve observed environment evidence.
    print(f"Stage {stage}; GPU {gpu_name}", flush=True)  # Show the verified hardware rather than an intention.
    frame = pd.read_parquet(PROCESSED / "dota_snapshots.parquet")  # Load only the prepared compact feature table.
    features = [column for column in frame if column.endswith("_diff")]  # Enforce the explicit feature allowlist.
    records, predictions = [], []  # Accumulate model measurements and per match probability evidence.
    for horizon in sorted(frame.horizon_seconds.unique()):  # Evaluate each prespecified observation time independently.
        observed = frame.loc[frame.horizon_seconds == horizon]  # Keep the horizon specific active match cohort.
        train = observed.loc[observed.partition == "train"]  # Only these labels may influence model weights.
        validation = observed.loc[observed.partition == "validation"]  # Use this period for checkpoint decisions and development reporting.
        calibration = observed.loc[observed.partition == "calibration"]  # Fit probability adjustment on a distinct period.
        for name in MODELS:  # Give every model the same eligible matches and feature columns.
            for seed in SEEDS:  # Repeat stochastic training to quantify seed variation.
                filename = MODEL_DIR / f"{name}_{int(horizon)}_{seed}.joblib"  # Give each fitted experiment a unique cache key.
                if stage == "development":  # Fit and inspect only development partitions in this stage.
                    if filename.exists():  # Reuse an unchanged completed fit when resuming an interrupted run.
                        bundle = joblib.load(filename)  # Preserve the same weights and calibration parameter.
                    else:  # Fit only when this experiment has not already completed.
                        model, seconds = fit_model(name, train[features], train.radiant_win.astype(int), validation[features], validation.radiant_win.astype(int), seed)  # Fit without test features or labels.
                        if name == "xgboost":  # Verify that boosting also used the requested GPU.
                            config = json.loads(model.get_booster().save_config())  # Inspect the fitted backend configuration.
                            assert "cuda" in config["learner"]["generic_param"]["device"]  # Fail rather than reporting CPU fallback as GPU training.
                        cp = model.predict_proba(calibration[features])[:, 1]  # Obtain predictions for calibrator fitting only.
                        temperature = temperature_fit(calibration.radiant_win.astype(int).to_numpy(), cp)  # Learn confidence adjustment on the separate period.
                        bundle = dict(model=model, training_seconds=seconds, temperature=temperature, features=features, horizon_seconds=int(horizon), seed=seed, model_name=name)  # Save everything needed for fixed later inference.
                        joblib.dump(bundle, filename)  # Checkpoint each completed experiment to avoid unnecessary reruns.
                    evaluation = validation  # Keep final test outcomes outside development evaluation.
                else:  # At this point the final protocol has already been frozen.
                    bundle = joblib.load(filename)  # Reuse fitted weights and calibration unchanged.
                    evaluation = observed.loc[observed.partition == "test"]  # Open the final period only in this deliberate stage.
                probability = bundle["model"].predict_proba(evaluation[features])[:, 1]  # Generate predictions without fitting on evaluation outcomes.
                adjusted = apply_temperature(probability, bundle["temperature"])  # Apply the calibration parameter learnt elsewhere.
                y = evaluation.radiant_win.astype(int).to_numpy()  # Access the eligible outcomes for the designated evaluation stage.
                for kind, p in [("raw", probability), ("temperature", adjusted)]:  # Report both rather than hiding an unhelpful calibration adjustment.
                    record = dict(horizon_seconds=int(horizon), model=name, seed=seed, probability=kind, matches=len(y), training_seconds=bundle["training_seconds"], temperature=bundle["temperature"], **metrics(y, p))  # Save direct measurements with their experimental identity.
                    if stage == "final":  # Quantify match sample uncertainty within the final evaluation.
                        record["match_bootstrap_95_interval"] = bootstrap_interval(y, p)  # Keep this distinct from training seed spread.
                    records.append(record)  # Preserve every model and seed, including disappointing runs.
                predictions.append(pd.DataFrame(dict(match_id=evaluation.match_id.to_numpy(), horizon_seconds=horizon, model=name, seed=seed, y=y, raw=probability, temperature=adjusted)))  # Retain exact predictions underlying tables and calibration plots.
                print(f"{stage}: {horizon // 60} min {name} seed {seed}; AUC {metrics(y, probability)['auc']:.4f}; training {bundle['training_seconds']:.1f}s", flush=True)  # Provide concise observable progress.
    output = pd.concat(predictions, ignore_index=True)  # Combine the per experiment prediction tables.
    output.to_parquet(PRED_DIR / f"D_{stage}_predictions.parquet", index=False)  # Enable later analyses without touching model training again.
    save_json(f"D_{'final' if stage == 'final' else 'development'}_metrics.json", {"stage": stage, "records": records, "features": features, "seeds": SEEDS, "note": "Intervals resample matches for fixed fitted predictions; seed spread is reported separately"})  # Record the full evidence rather than a manually typed summary.
    if stage == "final":  # Mark completion so later notebooks use the saved predictions.
        protocol["status"] = "final evaluation complete; do not tune on these results"  # Preserve the boundary between development and final reporting.
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")  # Save the state transition alongside the split record.
    return records  # Let notebooks display measurements without duplicating implementation.


if __name__ == "__main__":  # Command line execution is separate from importing helpers in a notebook.
    parser = argparse.ArgumentParser()  # Document the allowed run stages.
    parser.add_argument("--stage", choices=["development", "final"], default="development")  # Default to the stage that keeps test locked.
    arguments = parser.parse_args()  # Read the chosen stage from the command line.
    run(arguments.stage)  # Execute the shared workflow.
