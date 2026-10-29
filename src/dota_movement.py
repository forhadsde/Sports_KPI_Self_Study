"""Measure sampling sensitivity on matched trajectory endpoints.

This is a measurement study on development matches, not a causal experiment.
Distances are in recorded map coordinate units, not metres. Native sampling is
not continuous ground truth. Death, gaps and unusually large jumps split paths.
"""
import json  # Save measured summaries and operational assumptions.
import numpy as np  # Calculate displacement and matched path lengths.
import pandas as pd  # Group observations into player trajectories.
import pyarrow.parquet as pq  # Read only required columns and game times.
from dota_data import PROCESSED, RAW, ROOT, save_json  # Reuse project paths and evidence saving.


def trajectory_pairs(group, speed_multiplier=1.75):
    """Compare coarse and native paths over identical endpoints per segment.

    Sampling at a minute grid uses the most recent preceding observation, not
    a future value. Repeating grid offsets measures phase sensitivity.
    """
    group = group.sort_values("game_time").drop_duplicates("game_time")  # Establish ordered unique observations within one player and match.
    t = group.game_time.to_numpy()  # Extract the observed seconds.
    x, y = group.x.to_numpy(), group.y.to_numpy()  # Extract recorded positions in map coordinate units.
    alive = group.is_alive.to_numpy(dtype=bool)  # Do not connect living movement through death and respawn.
    speed = group.move_speed.to_numpy(dtype=float)  # Use reported movement speed to flag unusually large steps.
    dt = np.diff(t, prepend=np.nan)  # Measure elapsed time since the preceding observation.
    distance = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))  # Calculate straight line displacement between successive sampled positions.
    previous_alive = np.r_[False, alive[:-1]]  # Mark whether the preceding record was alive.
    previous_speed = np.r_[speed[0], speed[:-1]]  # Use both endpoint speeds when allowing a step.
    valid = (dt > 0) & (dt <= 1.2) & alive & previous_alive & np.isfinite(distance) & (distance <= np.maximum(speed, previous_speed) * speed_multiplier * dt + 64)  # Exclude gaps, dead intervals and implausibly large local steps under the stated operational rule.
    segments = np.cumsum(~valid)  # Start a new path segment whenever we cannot justify connecting two points.
    output = []  # Collect matched comparisons for the four minute grid phases.
    for offset in [0, 15, 30, 45]:  # Avoid conclusions depending on a single arbitrary minute boundary.
        native_total, coarse_total, intervals = 0.0, 0.0, 0  # Accumulate only intervals with matched endpoints.
        targets = np.arange(offset, 1200.001, 60.0)  # Define a minute spaced grid within the fixed observation window.
        chosen = np.searchsorted(t, targets, side="right") - 1  # Choose preceding observations rather than future ones.
        eligible = chosen >= 0  # Remove targets before the first observed point.
        chosen, targets = chosen[eligible], targets[eligible]  # Keep matching arrays for the remaining targets.
        eligible = (targets - t[chosen] <= 1.2) & alive[chosen]  # Reject stale and dead observations on the grid.
        chosen = chosen[eligible]  # Retain the valid coarse observation indices.
        cumulative = np.cumsum(np.where(valid, distance, 0))  # Sum native displacements within justified segments.
        for start, end in zip(chosen[:-1], chosen[1:]):  # Compare consecutive eligible coarse samples.
            if segments[start] != segments[end]:  # Never draw a coarse line across a death, gap or jump boundary.
                continue  # Discard unmatched discontinuous intervals from both methods.
            native_total += cumulative[end] - cumulative[start]  # Follow all intervening native samples between these exact endpoints.
            coarse_total += float(np.hypot(x[end] - x[start], y[end] - y[start]))  # Draw the straight coarse line between the same endpoints.
            intervals += 1  # Record the amount of valid comparison support.
        output.append({"offset_seconds": offset, "native_distance": native_total, "coarse_distance": coarse_total, "matched_intervals": intervals})  # Return paired totals rather than incomparable whole path lengths.
    return output, int((~valid).sum()), len(group)  # Provide exclusion diagnostics alongside the comparisons.


def run():
    """Scan trajectories once and save match level comparisons for later plots."""
    path = PROCESSED / "dota_movement_pairs.parquet"  # Reuse the expensive trajectory scan when unchanged.
    partitions = pd.read_parquet(PROCESSED / "dota_partitions.parquet")  # Use the same frozen membership as prediction experiments.
    metadata = pd.read_parquet(RAW / "matches.parquet", columns=["match_id", "duration_sec"])  # Determine which matches are still active at the end of the window.
    eligible_ids = set(metadata.loc[metadata.duration_sec > 1200, "match_id"])  # Apply an observation eligibility rule, never a predictor.
    allowed = set(partitions.loc[partitions.partition.isin(["train", "validation"]), "match_id"]) & eligible_ids  # Keep final test trajectories and short completed matches outside this measurement cohort.
    if path.exists():  # A notebook rerun should not repeat every source read unnecessarily.
        pairs = pd.read_parquet(path)  # Load the previously generated paired measurements.
    else:  # Build the paired table only when it does not exist.
        records = []  # Accumulate per player records before match level aggregation.
        files = sorted((RAW / "ticks").glob("*.parquet"))  # Scan each original shard once in a stable order.
        columns = ["match_id", "slot", "tick", "game_time", "x", "y", "is_alive", "move_speed"]  # Retain raw ticks to join the independently audited match clock.
        for index, file in enumerate(files):  # Bound memory by processing one shard at a time.
            table = pq.read_table(file, columns=columns)  # Replay elapsed time cannot safely filter a match clock window when pauses occur.
            frame = table.to_pandas()  # Work on the selected portion of this shard.
            frame = frame.loc[frame.match_id.isin(allowed)]  # Drop final test and calibration matches before calculating summaries.
            clock = pd.read_parquet(PROCESSED / "clock_maps" / file.name)  # Use the same combat log clock bridge as prediction snapshots.
            aligned_parts = []  # Assemble correctly aligned state rows for this small shard.
            for match_id, states in frame.groupby("match_id", sort=False):  # Match clocks must never be joined across games.
                bridge = clock.loc[clock.match_id == match_id, ["tick", "match_clock"]].rename(columns={"tick": "anchor_tick"}).sort_values("anchor_tick")  # Keep the relevant event clock records in time order.
                aligned = pd.merge_asof(states.sort_values("tick"), bridge, left_on="tick", right_on="anchor_tick", direction="backward")  # Use only a preceding clock observation.
                aligned = aligned.loc[(aligned.tick - aligned.anchor_tick <= 36) & aligned.match_clock.between(0, 1200)].copy()  # Reject stale clock mappings and states outside the match time window.
                aligned["game_time"] = aligned.match_clock  # Explicitly replace misleading replay elapsed time with the audited match clock estimate.
                aligned_parts.append(aligned)  # Keep only the aligned states for movement analysis.
            frame = pd.concat(aligned_parts, ignore_index=True) if aligned_parts else frame.iloc[:0]  # Handle source shards containing no eligible development games.
            for (match_id, slot), group in frame.groupby(["match_id", "slot"], sort=False):  # Treat each player's aligned trajectory separately.
                values, breaks, observations = trajectory_pairs(group)  # Apply the same operational path rule to both sampling resolutions.
                for value in values:  # Keep all grid phases rather than selecting the most favourable one.
                    records.append({"match_id": int(match_id), "slot": int(slot), "segment_breaks": breaks, "source_observations": observations, **value})  # Retain an auditable player and match identity.
            if index % 50 == 0:  # Keep progress visible during CPU source scanning.
                print(f"Movement: {index + 1}/{len(files)} shards", flush=True)  # Report file progress, not an unsupported completion estimate.
        pairs = pd.DataFrame(records)  # Convert the recorded comparisons into a compact reusable table.
        pairs.to_parquet(path, index=False)  # Preserve the measurements without modifying raw source files.
    matches = pairs.groupby(["match_id", "offset_seconds"])[["native_distance", "coarse_distance", "matched_intervals"]].sum().reset_index()  # Use complete matches as the uncertainty unit.
    matches = matches.loc[matches.native_distance > 0].copy()  # Ratios are undefined when the native comparison distance is zero.
    matches["retained_fraction"] = matches.coarse_distance / matches.native_distance  # Express how much sampled displacement remains at coarse resolution.
    assert (matches.retained_fraction <= 1 + 1e-8).all()  # Matched endpoints obey the triangle inequality up to floating point error.
    summaries = []  # Summarise every offset separately.
    for offset, subset in matches.groupby("offset_seconds"):  # Do not pool four correlated offsets as independent matches.
        values = subset.retained_fraction.to_numpy()  # Extract one ratio per match for this grid phase.
        rng = np.random.default_rng(1234)  # Fix resampling for reproducible descriptive intervals.
        boot = [np.median(values[rng.integers(0, len(values), len(values))]) for _ in range(1000)]  # Resample entire matches for the median's uncertainty.
        summaries.append({"offset_seconds": int(offset), "matches": len(values), "median_retained_fraction": float(np.median(values)), "retained_fraction_iqr": np.quantile(values, [.25, .75]).tolist(), "median_95_match_bootstrap_interval": np.quantile(boot, [.025, .975]).tolist(), "matched_intervals": int(subset.matched_intervals.sum())})  # Save observed effect size, spread and support.
    matches.to_parquet(ROOT / "outputs/results/D_movement_match_pairs.parquet", index=False)  # Allow figures and checks without repeating source scans.
    save_json("D_movement.json", {"clock_alignment_version": "combat_event_anchor_v2", "window_seconds": [0, 1200], "native_sampling": "approximately one second with preceding combat event match clock estimates", "coarse_interval_seconds": 60, "offsets_seconds": [0, 15, 30, 45], "speed_multiplier": 1.75, "jump_slack_coordinate_units": 64, "max_gap_seconds": 1.2, "matched_player_offset_rows": len(pairs), "unique_development_matches": int(pairs.match_id.nunique()), "summaries": summaries, "interpretation": "Operational sampled path displacement on matched continuous segments, not true walking distance or causal influence on winning", "limits": ["Death and gaps split segments", "The speed based jump rule may exclude genuine rapid movement and miss some ability movement", "Native sampling is not continuous ground truth", "Event anchored clocks have finite sampling resolution", "No claim about the magnitude in League of Legends", "Offsets are correlated within matches", "Only training and validation matches active at the window end; no final test outcomes used"]})  # State measurement assumptions next to numerical results.
    return matches  # Return the paired table for notebook inspection.


if __name__ == "__main__":  # Importing helpers does not trigger a long source scan.
    run()  # Run the measurement experiment deliberately.
