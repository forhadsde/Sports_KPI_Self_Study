"""Shared helpers: paths, seeds and saving results."""
import json  # Save evidence in a portable labelled text format.
import random  # Control Python's own random number generator.
from pathlib import Path  # Build paths without assuming a particular working directory.

import numpy as np  # Control numerical random sampling used by the study.

ROOT = Path(__file__).resolve().parents[1]  # Locate the project from this helper's saved location.
DATA_RAW = ROOT / "data" / "raw"  # Preserve downloads separately from transformations.
DATA_PROC = ROOT / "data" / "processed"  # Cache derived tables without modifying source data.
FIG = ROOT / "outputs" / "figures"  # Keep visual evidence together.
RES = ROOT / "outputs" / "results"  # Keep machine readable measurements together.
MODELS = ROOT / "outputs" / "models"  # Store fitted weights locally to avoid unnecessary retraining.
SEEDS = [0, 1, 2]  # Define training repetitions once for all study modules.

for p in (DATA_RAW, DATA_PROC, FIG, RES, MODELS):  # Ensure each expected output location exists.
    p.mkdir(parents=True, exist_ok=True)  # Reuse existing directories without replacing their contents.


def set_seed(seed: int = 0) -> None:
    random.seed(seed)  # Repeat Python sampling when the same seed is chosen.
    np.random.seed(seed)  # Repeat NumPy sampling independently of Python's generator.
    try:  # Allow data only utilities to work even when PyTorch is not installed.
        import torch  # Import the optional training library only when setting its seed.
        torch.manual_seed(seed)  # Repeat CPU weight initialisation and sampling.
        torch.cuda.manual_seed_all(seed)  # Repeat GPU sampling on every available device.
    except ImportError:  # A data only environment does not need neural training support.
        pass  # Leave Python and NumPy seeded without inventing a GPU dependency.


def save_result(name: str, obj: dict) -> Path:
    """Save a labelled result; Dota modules also use dota_data.save_json."""
    path = RES / f"{name}.json"  # Give the measurement a stable evidence filename.
    with open(path, "w", encoding="utf-8") as f:  # Preserve readable text consistently across platforms.
        json.dump(obj, f, indent=2, default=float)  # Serialise numerical scalars without losing the field labels.
    return path  # Let the caller link the saved evidence in a notebook.
