"""Models and measurements shared by the guided Dota 2 notebooks.

The neural architecture is inspired by Hojaji and colleagues, not claimed as
an exact reproduction. All estimators expose probabilities for the same target.
"""
import copy  # Keep a separate copy of the best network weights.
import random  # Control Python's random generator for repeatable runs.
import time  # Measure elapsed training time without using wall clock dates.

import numpy as np  # Store feature matrices and probability arrays.
import torch  # Train and evaluate the neural network on the GPU.
from scipy.optimize import minimize_scalar  # Fit one calibration parameter on a separate partition.
from scipy.special import expit, logit  # Convert safely between probabilities and log odds.
from sklearn.ensemble import RandomForestClassifier  # Provide a nonlinear tree baseline.
from sklearn.linear_model import LogisticRegression  # Provide a simple linear baseline.
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score  # Measure complementary aspects of predictions.
from sklearn.pipeline import make_pipeline  # Fit scaling and a model together without leaking data.
from sklearn.preprocessing import StandardScaler  # Put neural and linear inputs on comparable scales.
from xgboost import XGBClassifier  # Provide a boosted tree comparison with CUDA support.


def seed_everything(seed):
    """Control randomness; repeated seeds measure variation, not independent datasets."""
    random.seed(seed)  # Fix Python's generator.
    np.random.seed(seed)  # Fix NumPy's generator.
    torch.manual_seed(seed)  # Fix CPU and common PyTorch random draws.
    torch.cuda.manual_seed_all(seed)  # Fix every CUDA device's random draws.
    torch.backends.cudnn.benchmark = False  # Avoid choosing convolution algorithms by timing.
    torch.use_deterministic_algorithms(True)  # Fail rather than silently using a known nondeterministic operation.


def metrics(y, probability):
    """Return discrimination, threshold accuracy and two probability losses.

    Higher AUC and accuracy are better. Lower Brier score and log loss are
    better. Neither loss isolates calibration from discrimination by itself.
    """
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)  # Avoid undefined logarithms at exact zero or one.
    return {"auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p >= .5)), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}  # Keep a fixed threshold and explicit label ordering.


def temperature_fit(y, probability):
    """Fit temperature on calibration data, never on the final test data.

    A temperature above one softens probabilities towards one half. A value
    below one makes them more extreme. This cannot repair incorrect ranking.
    """
    scores = logit(np.clip(probability, 1e-7, 1 - 1e-7))  # Work in log odds so one scalar changes confidence.
    result = minimize_scalar(lambda log_temperature: log_loss(y, expit(scores / np.exp(log_temperature)), labels=[0, 1]), bounds=(np.log(.25), np.log(4)), method="bounded")  # Fit a positive, bounded temperature using calibration outcomes only.
    return float(np.exp(result.x))  # Return an ordinary positive temperature.


def apply_temperature(probability, temperature):
    """Apply a fitted temperature without looking at evaluation outcomes."""
    return expit(logit(np.clip(probability, 1e-7, 1 - 1e-7)) / temperature)  # Preserve ranking while adjusting confidence.


class NeuralPredictor:
    """A small two hidden layer network with training only standardisation.

    Inputs are numeric team differences. Outputs are Radiant win probabilities.
    The validation period chooses an epoch; calibration and test remain separate.
    """
    def __init__(self, seed=0):
        self.seed = seed  # Remember the training seed in the saved estimator.
        self.device = "cuda"  # Require the requested GPU instead of quietly falling back.

    def fit(self, X, y, X_validation, y_validation):
        seed_everything(self.seed)  # Make this particular training run reproducible.
        assert torch.cuda.is_available(), "CUDA is required for this study's neural runs."  # Report a genuine GPU failure clearly.
        torch.set_num_threads(4)  # Avoid excessive CPU thread overhead for a small dataset.
        self.scaler = StandardScaler().fit(X)  # Estimate scaling from training features only.
        x = torch.as_tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)  # Move the scaled training matrix to GPU memory.
        target = torch.as_tensor(np.asarray(y).copy(), dtype=torch.float32, device=self.device).reshape(-1, 1)  # Give binary labels a writable buffer and the same shape as network logits.
        xv = torch.as_tensor(self.scaler.transform(X_validation), dtype=torch.float32, device=self.device)  # Apply existing scaling to validation features.
        yv = torch.as_tensor(np.asarray(y_validation).copy(), dtype=torch.float32, device=self.device).reshape(-1, 1)  # Keep validation labels in a writable buffer separate from gradient updates.
        self.network = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(.5), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Dropout(.5), torch.nn.Linear(64, 1)).to(self.device)  # Use a documented architecture inspired by the original paper.
        optimizer = torch.optim.Adam(self.network.parameters(), lr=.001, weight_decay=1e-5)  # Use fixed settings rather than searching on test outcomes.
        loss_function = torch.nn.BCEWithLogitsLoss()  # Combine sigmoid and binary loss in a numerically stable form.
        best_loss, stale = float("inf"), 0  # Track improvements for early stopping.
        self.history = []  # Save learning curves instead of reporting only a final score.
        for epoch in range(200):  # Cap training at the prespecified number of passes.
            self.network.train()  # Enable dropout during training.
            order = torch.randperm(len(x), device=self.device)  # Shuffle training examples using the fixed random seed.
            total = 0.0  # Accumulate a sample weighted training loss.
            for batch in order.split(512):  # Keep small batches without repeatedly transferring data from the CPU.
                optimizer.zero_grad(set_to_none=True)  # Clear gradients from the previous update.
                loss = loss_function(self.network(x[batch]), target[batch])  # Compare predictions with training labels.
                loss.backward()  # Calculate how each weight affects the loss.
                optimizer.step()  # Move weights in a direction that reduces training loss.
                total += float(loss.detach()) * len(batch)  # Record loss without retaining the computation graph.
            self.network.eval()  # Disable dropout for comparable validation measurements.
            with torch.no_grad():  # Validation must not change the weights.
                validation_loss = float(loss_function(self.network(xv), yv))  # Use validation only to choose when to stop.
            self.history.append({"epoch": epoch + 1, "training_loss": total / len(x), "validation_loss": validation_loss})  # Preserve the observed curve.
            if validation_loss < best_loss - 1e-5:  # Require a small genuine improvement to reset patience.
                best_loss, stale = validation_loss, 0  # Update the best score and clear the no improvement count.
                best_weights = copy.deepcopy(self.network.state_dict())  # Preserve a copy independent of future updates.
                self.best_epoch = epoch + 1  # Record the selected checkpoint for the report.
            else:  # An unchanged or worse validation score consumes patience.
                stale += 1  # Count consecutive epochs without sufficient improvement.
            if stale >= 25:  # Stop using validation, never final test, when progress stalls.
                break  # End training without spending unnecessary GPU time.
        self.network.load_state_dict(best_weights)  # Restore the selected checkpoint rather than the last epoch.
        self.network.eval()  # Leave the saved model ready for deterministic inference.
        return self  # Match the usual estimator interface.

    def predict_proba(self, X):
        self.network.eval()  # Ensure dropout stays disabled for predictions.
        with torch.no_grad():  # Predictions do not require gradients.
            x = torch.as_tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)  # Reuse training scaling and move inputs to CUDA.
            p = torch.sigmoid(self.network(x)).cpu().numpy().ravel()  # Convert logits into CPU probability values.
        return np.column_stack([1 - p, p])  # Match scikit learn's negative and positive class probability convention.


def fit_model(name, X, y, X_validation, y_validation, seed):
    """Fit one fixed model and return it with elapsed training seconds."""
    started = time.perf_counter()  # Start timing immediately before model construction and fitting.
    if name == "logistic":  # A linear decision rule is a useful simple comparison.
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=seed))  # Scale using training data within the pipeline.
    elif name == "random_forest":  # Trees can model nonlinear relationships without standardisation.
        model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, max_features="sqrt", n_jobs=4, random_state=seed)  # Use fixed complexity and controlled parallelism.
    elif name == "xgboost":  # Boosting adds trees sequentially to reduce remaining error.
        model = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_lambda=1, tree_method="hist", device="cuda", n_jobs=4, random_state=seed, eval_metric="logloss")  # Request GPU histogram training explicitly.
    elif name == "neural_network":  # The adapted neural model uses validation for checkpoint selection.
        model = NeuralPredictor(seed).fit(X, y, X_validation, y_validation)  # Train on GPU and retain the best validation epoch.
        torch.cuda.synchronize()  # Ensure timing includes completed CUDA operations.
        return model, time.perf_counter() - started  # Return a fitted model and observed elapsed time.
    else:  # A misspelled model name must not select an arbitrary fallback.
        raise ValueError(name)  # Stop with the unexpected value visible.
    model.fit(X, y)  # Fit the selected classical estimator using training data only.
    if name == "xgboost":  # GPU work can finish asynchronously.
        torch.cuda.synchronize()  # Wait before ending the timing interval.
    return model, time.perf_counter() - started  # Report the estimator and its actual training time.
