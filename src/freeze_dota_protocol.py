"""Record the analysis decision before a fresh run opens final test outcomes."""
import argparse  # Require a deliberate acknowledgement of development review.
import hashlib  # Identify the implementation used at the decision boundary.
import json  # Preserve a readable protocol alongside results.
from dota_data import ROOT  # Use the same project location as the experiment.


def freeze():
    result = ROOT / "outputs/results"  # Locate evidence from the completed development stages.
    if (result / "D_final_metrics.json").exists():  # Never retrospectively rewrite an evaluated protocol.
        raise RuntimeError("Final evaluation already exists; the frozen protocol must remain unchanged.")  # Protect the historical record.
    for name in ["D_development_metrics.json", "D_explanations.json", "D_refit_stability.json", "D_movement.json", "D_integrity_validation.json"]:  # Require actual development evidence before finalisation.
        assert (result / name).exists(), name  # Stop if a planned development stage is missing.
    path = result / "D_protocol.json"  # Reuse the partition record created before model fitting.
    protocol = json.loads(path.read_text())  # Preserve membership and date boundaries.
    protocol.update(analysis_frozen=True, primary_horizon_seconds=1200,
                    primary_comparison="logistic versus neural_network; raw probabilities; seed averaged predictions for paired intervals",
                    calibration_reporting="Always report raw and temperature adjusted probabilities; no choice using final outcomes",
                    bootstrap_reporting="1000 match resamples and UTC day block sensitivity for primary paired comparison",
                    matched_cohort_reporting="Neural model on common final matches observed at all horizons",
                    conditional_hero_extension="Deferred: prospective role provenance and generic inventory values unresolved")  # State decisions without using final outcomes.
    protocol["code_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["src/dota_data.py", "src/dota_models.py", "src/run_dota_study.py"]}  # Fingerprint the current implementation for a new run.
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")  # Save the deliberate boundary before evaluation.
    print("Protocol frozen; final evaluation is now permitted once.")  # Make the state transition visible.


if __name__ == "__main__":  # Importing this file cannot unlock the test set.
    parser = argparse.ArgumentParser()  # Explain the required review acknowledgement.
    parser.add_argument("--development-reviewed", action="store_true", required=True)  # Require an explicit command line action.
    parser.parse_args()  # Reject an omitted acknowledgement before writing anything.
    freeze()  # Record the reviewed protocol for a fresh experiment.
