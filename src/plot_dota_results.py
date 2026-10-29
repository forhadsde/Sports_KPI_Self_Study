"""Generate publication figures from saved results, without retraining models."""
import json  # Read the exact numerical evidence used in the report.
import joblib  # Read saved learning curves from the original fitted network.
import numpy as np  # Calculate fixed probability bins and plotting coordinates.
import pandas as pd  # Organise saved predictions and feature rankings.
import matplotlib  # Select a noninteractive renderer for reproducible exported figures.
matplotlib.use("Agg")  # Save figures without requiring a visible desktop window.
import matplotlib.pyplot as plt  # Draw standard scientific plots.
from dota_data import PROCESSED, ROOT, save_json  # Share project paths and traceable result saving.

NAMES = {"logistic": "Logistic regression", "random_forest": "Random Forest", "xgboost": "XGBoost", "neural_network": "Neural network"}  # Use consistent model names across artifacts.
COLOURS = {"logistic": "#2369a4", "random_forest": "#178268", "xgboost": "#bd6b20", "neural_network": "#8551a1"}  # Keep model identities readable in every figure.
LABELS = {"net_worth_diff": "Net worth", "xp_diff": "Experience", "kills_diff": "Kills", "assists_diff": "Assists", "last_hits_diff": "Last hits", "denies_diff": "Denies", "level_diff": "Levels", "hp_diff": "Health", "is_alive_diff": "Players alive", "towers_alive_diff": "Towers alive"}  # Explain technical feature names in ordinary words.
FIG = ROOT / "outputs/figures"  # Store exported figures separately from tables and fitted weights.


def save(figure, name):
    FIG.mkdir(parents=True, exist_ok=True)  # Create the figure folder on a fresh checkout.
    figure.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")  # Export a sharp image suitable for notebooks and a short report.
    figure.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")  # Retain vector graphics for clear report printing.
    plt.close(figure)  # Release memory instead of leaving many figure windows open.


def run():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .16})  # Use consistent legible styling with unobtrusive grids.
    summary = json.loads((ROOT / "outputs/results/D_summary.json").read_text())  # Load the generated summary rather than manually entering values.
    table = pd.DataFrame(summary["test_table"])  # Extract all model and seed aggregate metrics.
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))  # Compare discrimination and probability error in adjacent panels.
    for name, label in NAMES.items():  # Include every agreed model rather than only the strongest.
        raw = table.loc[(table.model == name) & (table.probability == "raw")].sort_values("horizon_seconds")  # Keep raw predictions in observation time order.
        adjusted = table.loc[(table.model == name) & (table.probability == "temperature")].sort_values("horizon_seconds")  # Keep calibrated losses separate.
        axes[0].errorbar(raw.horizon_seconds / 60, raw.auc_mean, yerr=raw.auc_sd, marker="o", color=COLOURS[name], label=label, capsize=3)  # Show mean AUC and training seed spread.
        axes[1].plot(raw.horizon_seconds / 60, raw.brier_mean, "o-", color=COLOURS[name])  # Raw Brier score uses a solid line.
        axes[1].plot(adjusted.horizon_seconds / 60, adjusted.brier_mean, "s:", color=COLOURS[name])  # Temperature adjusted Brier score uses a dotted line.
    axes[0].set(ylabel="ROC AUC", ylim=(.5, 1), title="Ranking eventual winners")  # AUC measures ranking rather than classification accuracy.
    axes[1].set(ylabel="Brier score", ylim=(0, .3), title="Probability error: solid raw, dotted calibrated")  # Smaller Brier score is better.
    for axis in axes:  # Use the same fixed observation times on both panels.
        axis.set(xlabel="Observation time in match minutes", xticks=[10, 20, 30])  # Cohort sizes are reported separately because later matches must still be active.
    axes[0].legend(frameon=False, fontsize=8)  # Name the model colours without obscuring the curves.
    figure.tight_layout()  # Prevent overlapping labels before export.
    save(figure, "D_performance")  # Save the figure from measured values.
    average = pd.read_parquet(ROOT / "outputs/results/D_seed_averaged_predictions.parquet")  # Calibration plots use the prespecified seed averaged probabilities.
    primary = average.loc[average.horizon_seconds == 1200]  # Plot the primary observation time selected before final evaluation.
    figure, axes = plt.subplots(2, 2, figsize=(7.5, 6.4), sharex=True, sharey=True)  # Give each model a readable calibration panel.
    bins = []  # Preserve every plotted bin and its support count as numerical evidence.
    for axis, (name, label) in zip(axes.ravel(), NAMES.items()):  # Reuse the same model order as the performance plot.
        subset = primary.loc[primary.model == name]  # Keep one prediction per eligible test match.
        axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)  # The diagonal represents perfect descriptive calibration.
        for kind, colour, marker in [("raw", "#2369a4", "o"), ("temperature", "#bd6b20", "s")]:  # Show both versions even if adjustment does not help.
            assignment = np.minimum((subset[kind].to_numpy() * 10).astype(int), 9)  # Use ten fixed equal width bins without tuning on outcomes.
            points = []  # Collect nonempty bins for a connected descriptive curve.
            for bucket in range(10):  # Keep the bin edges fixed across models.
                group = subset.loc[assignment == bucket]  # Select the predictions in one interval.
                if group.empty:  # Empty intervals have no observed outcome frequency.
                    continue  # Do not invent a point at zero.
                record = dict(model=name, probability=kind, bin=bucket, matches=len(group), mean_prediction=float(group[kind].mean()), win_fraction=float(group.y.mean()))  # Record both probability and observed frequency.
                bins.append(record)  # Make plotted numbers independently inspectable.
                points.append(record)  # Retain bin order for the curve.
            axis.plot([p["mean_prediction"] for p in points], [p["win_fraction"] for p in points], marker=marker, color=colour, label="Raw" if kind == "raw" else "Temperature", linewidth=1)  # Draw a descriptive curve without implying independent bins prove causation.
        axis.set(title=label, xlim=(0, 1), ylim=(0, 1))  # Use identical scales for an honest comparison.
    axes[0, 0].legend(frameon=False, fontsize=8)  # Explain the probability versions once.
    for axis in axes[1]:  # Label the bottom row without redundant clutter.
        axis.set_xlabel("Mean predicted win probability")  # Explain the horizontal quantity in plain language.
    for axis in axes[:, 0]:  # Label the left column of shared axes.
        axis.set_ylabel("Observed fraction of wins")  # Calibration compares predicted and observed frequencies.
    figure.tight_layout()  # Keep labels outside neighbouring panels.
    save(figure, "D_calibration")  # Export the primary calibration figure.
    save_json("D_calibration_bins.json", {"horizon_seconds": 1200, "equal_width_bins": 10, "prediction_aggregation": "Mean across fixed seed models", "records": bins, "limits": "Descriptive bin curves; sparse bins and dependent matches limit interpretation"})  # Store all calibration plot values with their scope.
    ranks = pd.DataFrame(summary["development_rank_summary"])  # Explanations use development observations rather than final test selection.
    ranks = ranks.loc[ranks.horizon_seconds == 1200]  # Use the same primary time for the global explanation panel.
    refits = pd.DataFrame(summary["training_refit_rank_summary"])  # Refitting measures a different source of uncertainty.
    order = ranks.groupby("feature").mean_rank.mean().sort_values().index.tolist()  # Order features by average development rank for readable comparison.
    matrices = [ranks.pivot(index="feature", columns="model", values="mean_rank").reindex(index=order, columns=NAMES), refits.pivot(index="feature", columns="model", values="top3_frequency").reindex(index=order, columns=NAMES)]  # Align model and feature labels exactly.
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)  # Place nominal rank beside training sample stability.
    for axis, matrix, title, cmap, low, high in zip(axes, matrices, ["Mean development rank", "Top three frequency across training refits"], ["viridis_r", "YlGnBu"], [1, 0], [10, 1]):  # Make the two colour scales explicit.
        image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=low, vmax=high)  # Draw observed ranks or inclusion fractions.
        axis.set(title=title, xticks=range(4), xticklabels=["Logistic", "Forest", "XGBoost", "Neural"], yticks=range(len(order)), yticklabels=[LABELS[f] for f in order])  # Translate feature names into readable labels.
        axis.grid(False)  # Gridlines would obscure the colour cells.
        for row in range(len(order)):  # Add exact rounded values so the heatmap is not only qualitative.
            for column in range(4):  # Annotate each model and feature combination.
                value = matrix.iloc[row, column]  # Read the measured value from the aligned matrix.
                axis.text(column, row, f"{value:.1f}" if high == 10 else f"{value:.2f}", ha="center", va="center", color="white" if (high == 10 and value > 6) or (high == 1 and value > .6) else "black", fontsize=8)  # Keep text readable against the actual light and dark cells.
        figure.colorbar(image, ax=axis, shrink=.8)  # Show that rank and frequency are different numerical quantities.
    figure.tight_layout()  # Avoid cutting off feature labels.
    save(figure, "D_rank_stability")  # Save both explanation views together.
    movement = pd.read_parquet(ROOT / "outputs/results/D_movement_match_pairs.parquet")  # Read paired distances already measured on development trajectories.
    figure, axis = plt.subplots(figsize=(6.5, 3.7))  # Keep this conditional measurement result visually separate from win prediction.
    offsets = [0, 15, 30, 45]  # Show every prespecified coarse grid phase.
    arrays = [movement.loc[movement.offset_seconds == offset, "retained_fraction"].to_numpy() * 100 for offset in offsets]  # Convert recorded fractions into percentages for readability.
    boxes = axis.boxplot(arrays, tick_labels=[f"{offset}s\nn = {len(values)}" for offset, values in zip(offsets, arrays)], showfliers=False, patch_artist=True)  # Disclose the valid matched cohort for each offset.
    for box in boxes["boxes"]:  # Keep one neutral colour because offsets are not separate model families.
        box.set_facecolor("#a6c9df")  # Make the interquartile range visible without implying a causal category.
    common = summary["movement"]["common_offset_medians"]  # Compare offsets on identical matches as a population sensitivity check.
    axis.plot(range(1, 5), [common[str(offset)] * 100 for offset in offsets], "D", color="#bd6b20", label=f"Common cohort medians (n = {summary['movement']['common_offset_matches']})")  # Distinguish changing cohort medians from the matched population result.
    axis.set(ylabel="Coarse distance / native distance (%)", xlabel="Offset of the minute sampling grid", ylim=(0, 105), title="Matched continuous segments only")  # Do not imply that these ratios describe complete matches or true movement.
    axis.legend(frameon=False, fontsize=8)  # Explain the common cohort marker.
    figure.tight_layout()  # Keep support counts and labels readable.
    save(figure, "D_movement")  # Export the measurement sensitivity figure.
    neural = joblib.load(ROOT / "outputs/models/neural_network_1200_0.joblib")["model"]  # Inspect an already fitted checkpoint, not a new training run.
    history = pd.DataFrame(neural.history)  # Read the actual saved learning curve.
    figure, axis = plt.subplots(figsize=(6.5, 3.2))  # Provide a simple educational learning curve in the notebook.
    axis.plot(history.epoch, history.training_loss, label="Training loss")  # Show optimisation on the training period.
    axis.plot(history.epoch, history.validation_loss, label="Validation loss")  # Show the period used for checkpoint selection.
    axis.axvline(neural.best_epoch, linestyle=":", color="0.35", label="Selected epoch")  # Explain why the final weights do not necessarily come from the last epoch.
    axis.set(xlabel="Training epoch", ylabel="Binary cross entropy", title="Reference network, primary observation time, seed zero")  # Name the exact experiment represented.
    axis.legend(frameon=False)  # Make the two data roles clear to a beginner.
    figure.tight_layout()  # Avoid overlapping axis text.
    save(figure, "D_learning_curve")  # Save a diagnostic figure without retraining.
    figure, axis = plt.subplots(figsize=(4.2, 3.1))  # Create a readable single column report version of the AUC panel.
    for name, label in NAMES.items():  # Retain every model in the compact report plot.
        raw = table.loc[(table.model == name) & (table.probability == "raw")].sort_values("horizon_seconds")  # Use the same measured values as the full figure.
        axis.plot(raw.horizon_seconds / 60, raw.auc_mean, "o-", color=COLOURS[name], label=label, markersize=4)  # Show mean seed performance without claiming confidence from tiny seed bars.
    axis.set(xlabel="Observation time (match minutes)", ylabel="ROC AUC", xticks=[10, 20, 30], ylim=(.5, 1))  # Keep the full relevant AUC scale.
    axis.legend(frameon=False, fontsize=7)  # Keep labels legible after single column scaling.
    figure.tight_layout()  # Avoid clipping a compact axis label.
    save(figure, "D_auc_report")  # Export a report specific rendering of existing measurements.
    figure, axis = plt.subplots(figsize=(4.2, 3.2))  # Make the primary comparison's calibration curve readable in one column.
    axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)  # Retain the perfect calibration reference.
    for name, kind, label, colour, marker in [("logistic", "raw", "Logistic raw", "#2369a4", "o"), ("neural_network", "raw", "Neural raw", "#8551a1", "o"), ("neural_network", "temperature", "Neural temperature", "#bd6b20", "s")]:  # Focus the report figure on the comparison fixed before final evaluation.
        rows = [b for b in bins if b["model"] == name and b["probability"] == kind]  # Reuse the saved fixed bins, not new outcome chosen intervals.
        axis.plot([b["mean_prediction"] for b in rows], [b["win_fraction"] for b in rows], marker=marker, color=colour, label=label, markersize=3, linewidth=1)  # Draw the already recorded descriptive curve.
    axis.set(xlabel="Mean predicted win probability", ylabel="Observed fraction of wins", xlim=(0, 1), ylim=(0, 1))  # Use the same probability scale on both axes.
    axis.legend(frameon=False, fontsize=7)  # Explain the three curves without a large panel title.
    figure.tight_layout()  # Keep all text within the compact exported figure.
    save(figure, "D_calibration_report")  # Export the report sized primary calibration view.
    figure, axis = plt.subplots(figsize=(4.2, 3.1))  # Make the movement measurement readable at a report column width.
    boxes = axis.boxplot(arrays, tick_labels=[str(o) for o in offsets], showfliers=False, patch_artist=True)  # Reuse every measured grid phase in the compact view.
    for box in boxes["boxes"]:  # Apply the same visual meaning as the full notebook figure.
        box.set_facecolor("#a6c9df")  # Blue boxes show the conditional match ratio distribution.
    axis.plot(range(1, 5), [common[str(o)] * 100 for o in offsets], "D", color="#bd6b20", markersize=4, label="Common cohort median")  # Control for changing cohort membership when comparing grid phases.
    axis.set(xlabel="Minute grid offset (seconds)", ylabel="Distance retained (%)", ylim=(0, 105))  # The caption defines matched endpoints and the native reference.
    axis.legend(frameon=False, fontsize=7)  # Keep the compact figure self explanatory.
    figure.tight_layout()  # Prevent clipped axis labels after scaling.
    save(figure, "D_movement_report")  # Save the report version without recalculating distances.
    print("Saved figures and calibration bin evidence", flush=True)  # Report artifact generation completion.


if __name__ == "__main__":  # Importing labels should not regenerate all figures.
    run()  # Build the figures deliberately from saved evidence.
