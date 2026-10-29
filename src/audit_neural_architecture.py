"""Describe the actual fitted neural reference without retraining it."""
from pathlib import Path  # Find the cached model inside the study folder.
import json  # Save inspectable architecture evidence for the methodology slides.
import joblib  # Read the fitted estimator already used in the completed experiment.
import torch  # Identify its actual linear layers and count their parameters.

ROOT = Path(__file__).resolve().parents[1]  # Resolve the project independently of the terminal folder.


def run():
    path = ROOT / "outputs/models/neural_network_1200_0.joblib"  # Inspect one prespecified fitted reference, not a new optimised model.
    bundle = joblib.load(path)  # Load existing weights without fitting or generating new predictions.
    model = bundle["model"]  # Extract the neural estimator from its calibration and provenance bundle.
    layers = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]  # Read the dimensions from actual trained layers.
    record = dict(source="Fitted primary neural reference, seed zero", stage="Post study architecture description", input_features=len(bundle["features"]), hidden_units=[layer.out_features for layer in layers[:-1]], output_units=layers[-1].out_features, total_hidden_neurons=sum(layer.out_features for layer in layers[:-1]), trainable_parameters=sum(parameter.numel() for parameter in model.network.parameters() if parameter.requires_grad), activation="ReLU in hidden layers; sigmoid applied to the output logit for probability inference", dropout_probabilities=[layer.p for layer in model.network if isinstance(layer, torch.nn.Dropout)], feature_names=bundle["features"], architecture_search_performed=False, note="The width is a fixed reference choice, not an empirically established optimum. No retraining or final prediction evaluation was performed by this audit.")  # Keep measured widths separate from their methodological interpretation.
    record["selected_epoch"] = model.best_epoch  # Describe the validation selected checkpoint from this actual fitted run.
    record["epochs_run"] = len(model.history)  # Distinguish completed training passes from the maximum allowed.
    output = ROOT / "outputs/results/D_neural_architecture_audit.json"  # Give the description a stable evidence filename.
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")  # Save exact counts instead of typing a guessed neuron total.
    print(json.dumps(record, indent=2))  # Display the fitted architecture transparently.
    return record  # Let the slide builder use the same recorded values.


if __name__ == "__main__":  # Importing the helper must not load model weights automatically.
    run()  # Describe the completed model only when deliberately executed.
