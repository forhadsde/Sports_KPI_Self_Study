"""Build time observable team statistics for the independent Dota 2 study.

Author: Md Forhadul Islam.
The dataset belongs to wolframko and its credited upstream providers.
All features are Radiant minus Dire differences, not final match totals.
"""
import hashlib  # Fingerprints make it possible to check the exact partition later.
import json  # Save the experimental settings in a readable machine format.
import re  # Match documented internal and display hero names despite spaces and underscores.
from pathlib import Path  # Build portable paths instead of joining Windows strings.

import numpy as np  # Arrays help us handle times and differences efficiently.
import pandas as pd  # DataFrames represent rows of match statistics.
import pyarrow.parquet as pq  # Read selected columns from compressed source tables.

ROOT = Path(__file__).resolve().parents[1]  # Find the project from this script's location.
RAW = ROOT / "data/raw/betty-dota2"  # Keep the original downloaded files separate.
PROCESSED = ROOT / "data/processed"  # Store derived tables without modifying originals.
RESULTS = ROOT / "outputs/results"  # Record every reported measurement and setting.
HORIZONS = [600, 1200, 1800]  # Observe matches at fixed seconds, not fractions of final duration.
STATS = ["net_worth", "xp", "kills", "assists", "last_hits", "denies", "level", "hp", "is_alive"]  # Exclude identities, labels and eventual duration.


def save_json(name, value):
    """Save a result next to the other evidence files and return its path."""
    RESULTS.mkdir(parents=True, exist_ok=True)  # Create the results folder if absent.
    path = RESULTS / name  # Keep a stable name for downstream notebooks.
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")  # Preserve readable text and dates.
    return path  # Let callers link to the exact generated file.


def make_partitions(metadata):
    """Assign complete UTC dates to chronological partitions before modelling.

    Date grouping avoids separating games on one day. It is not a guarantee
    that a tournament series never crosses midnight. Series IDs are absent.
    """
    target = PROCESSED / "dota_partitions.parquet"  # Reuse a frozen split rather than redrawing it.
    if target.exists():  # A later notebook must keep the same match assignments.
        return pd.read_parquet(target)  # Read the already frozen membership.
    ordered = metadata[["match_id", "start_time"]].sort_values(["start_time", "match_id"]).copy()  # Order without using outcomes.
    ordered["utc_day"] = pd.to_datetime(ordered.start_time, unit="s", utc=True).dt.floor("D")  # Keep a complete day together.
    boundaries = [ordered.iloc[int(len(ordered) * f)].utc_day for f in [.60, .75, .80]]  # Reserve separate tuning, calibration and final test periods.
    ordered["partition"] = np.select([ordered.utc_day < boundaries[0], ordered.utc_day < boundaries[1], ordered.utc_day < boundaries[2]], ["train", "validation", "calibration"], default="test")  # Assign each match to one time period.
    assert ordered.match_id.is_unique  # A match cannot have two competing assignments.
    ordered.to_parquet(target, index=False)  # Save membership so every model sees identical partitions.
    fingerprint = hashlib.sha256(target.read_bytes()).hexdigest()  # Detect an accidental change to the split.
    save_json("D_protocol.json", {"status": "development; final test locked", "horizons_seconds": HORIZONS, "seeds": [0, 1, 2], "partition_counts": ordered.partition.value_counts().to_dict(), "date_boundaries": list(map(str, boundaries)), "partition_sha256": fingerprint, "feature_rule": "Radiant minus Dire observed totals", "snapshot_tolerance_seconds": 1.2, "selection": "latest observation no later than horizon", "series_limitation": "No series IDs; complete UTC dates grouped; teams may recur", "excluded_inputs": ["eventual duration", "final scores", "final statistics", "retrospective roles", "match identifiers", "outcomes"], "planned_models": ["logistic", "random_forest", "xgboost", "neural_network"], "planned_primary_metrics": ["ROC AUC", "accuracy", "Brier score", "log loss"], "final_test_policy": "No final test predictions until development and protocol freeze are complete"})  # Record choices before fitting a model.
    return ordered  # Return membership without changing it later.


def latest_rows(path, columns, keys, anchors=None):
    """Read only rows immediately preceding each fixed observation time.

    Missing observations are excluded, never borrowed from the future.
    Parquet filtering reduces memory use, although file scanning still costs time.
    """
    if anchors is None:  # Tests may supply a small independent clock table.
        anchors = pd.read_parquet(PROCESSED / "dota_clock_cutoffs.parquet")  # Use combat log match time rather than the replay elapsed field.
    identifiers = pq.read_table(path, columns=["match_id"]).column("match_id").unique().to_pylist()  # Identify the matches in this source shard.
    anchors = anchors.loc[anchors.match_id.isin(identifiers)]  # Keep only cutoffs relevant to this shard.
    columns = list(dict.fromkeys(columns + ["tick"]))  # Include raw ticks so the temporal rule can be checked.
    filters = [[("match_id", "==", int(row.match_id)), ("tick", ">=", int(row.cutoff_tick) - 36), ("tick", "<", int(row.cutoff_tick))] for row in anchors.itertuples()]  # Strictly precede an event tick known to be no later than the requested match time.
    if not filters:  # Some shards may have no sufficiently recent event anchors at these horizons.
        return pd.DataFrame(columns=columns + ["horizon_seconds"])  # Return an empty compatible table rather than future observations.
    table = pq.read_table(path, columns=columns, filters=filters)  # Read the requested columns and eligible times.
    frame = table.to_pandas()  # Work on the small filtered table, not the entire source.
    output = []  # Collect one table for each time horizon.
    for horizon in HORIZONS:  # Keep time horizons separate so they cannot leak into each other.
        cutoff = anchors.loc[anchors.horizon_seconds == horizon, ["match_id", "cutoff_tick"]]  # Select this match time's safely preceding event tick.
        eligible = frame.merge(cutoff, on="match_id", validate="many_to_one")  # Attach the cutoff without using outcomes.
        eligible = eligible.loc[(eligible.tick < eligible.cutoff_tick) & (eligible.tick >= eligible.cutoff_tick - 36)].copy()  # Keep preceding states at most one native interval behind the safe anchor.
        eligible = eligible.sort_values("tick").drop_duplicates(keys, keep="last")  # Choose the last eligible raw state per player or building.
        eligible["horizon_seconds"] = horizon  # Preserve which observation time this row represents.
        output.append(eligible)  # Retain the rows for later team aggregation.
    return pd.concat(output, ignore_index=True)  # Combine horizons while retaining their labels.


def prepare_clocks():
    """Build a bridge between replay ticks and the combat log match clock.

    The publisher's game_time is replay elapsed time and can include pauses.
    Combat event timestamps agree with recorded match end times in the checked
    objective events. Their maximum per tick provides a conservative anchor.
    """
    directory = PROCESSED / "clock_maps"  # Cache the clock bridge for movement and snapshot extraction.
    directory.mkdir(parents=True, exist_ok=True)  # Keep derived clock maps separate from raw events.
    cutoff_path = PROCESSED / "dota_clock_cutoffs.parquet"  # Save only the small fixed horizon anchor table for modelling.
    if cutoff_path.exists():  # Avoid another full combat event scan on rerun.
        return pd.read_parquet(cutoff_path)  # Reuse the validated versioned clock bridge.
    cutoffs, drift = [], []  # Record usable anchors and the observed clock mismatch.
    files = sorted((RAW / "events").glob("*.parquet"))  # Scan combat logs where both clock representations are present.
    for index, path in enumerate(files):  # Bound memory by processing one source shard.
        events = pq.read_table(path, columns=["match_id", "tick", "game_time", "timestamp"]).to_pandas()  # Read timing fields without combat values or outcomes.
        clock = events.groupby(["match_id", "tick"], as_index=False).agg(match_clock=("timestamp", "max"), replay_time=("game_time", "first"))  # Use the latest logged game time at a replay tick.
        clock.to_parquet(directory / path.name, index=False)  # Retain the bridge so other analyses use the same clock interpretation.
        for horizon in HORIZONS:  # Define the fixed prediction times using match clock seconds.
            eligible = clock.loc[clock.match_clock.between(horizon - 1.2, horizon)].sort_values("tick").drop_duplicates("match_id", keep="last")  # Choose an event anchor that does not pass the desired time.
            cutoffs.extend({"match_id": int(row.match_id), "horizon_seconds": horizon, "cutoff_tick": int(row.tick), "anchor_match_time": float(row.match_clock), "anchor_replay_time": float(row.replay_time)} for row in eligible.itertuples())  # Preserve exact anchor evidence for audit.
        drift.extend((clock.replay_time - clock.match_clock).iloc[::100].tolist())  # Sample observed discrepancies without retaining every timing value in JSON.
        if index % 100 == 0:  # Report progress during the necessary source scan.
            print(f"Clock alignment: {index + 1}/{len(files)} shards", flush=True)  # Distinguish this audit from model training.
    result = pd.DataFrame(cutoffs)  # Collect horizon specific match anchors.
    result.to_parquet(cutoff_path, index=False)  # Save compact anchors for every subsequent model.
    save_json("D_clock_alignment.json", {"version": "combat_event_anchor_v2", "anchors": len(result), "matches": int(result.match_id.nunique()), "event_shards": len(files), "sampled_replay_minus_match_seconds_quantiles": np.quantile(drift, [0, .25, .5, .75, .95, 1]).tolist(), "anchor_max_age_match_seconds": 1.2, "state_rule": "raw tick strictly before latest safe event tick, within 36 replay ticks", "native_tick_rate": 30, "limitation": "Event anchored state timing has finite native resolution; it is not exact continuous game state"})  # Document the reason and residual approximation.
    return result  # Return anchors for direct inspection.


def prepare():
    """Build auditable snapshots; this function does not fit a model."""
    PROCESSED.mkdir(parents=True, exist_ok=True)  # Make the derived table folder inside the project.
    metadata = pd.read_parquet(RAW / "matches.parquet")  # Read outcomes and timing in a separate metadata table.
    partitions = make_partitions(metadata)  # Freeze match membership before creating any model.
    prepare_clocks()  # Verify match clock alignment before claiming fixed minute prediction.
    snapshot_path = PROCESSED / "dota_snapshots.parquet"  # Use a stable cache for expensive source scanning.
    if snapshot_path.exists():  # Avoid rescanning when a notebook is rerun unchanged.
        return pd.read_parquet(snapshot_path)  # Load the generated feature table.
    player_parts = []  # Accumulate compact observed player snapshots.
    files = sorted((RAW / "ticks").glob("*.parquet"))  # Enumerate source shards in a reproducible order.
    columns = ["match_id", "slot", "hero", "game_time", *STATS]  # Hero identity is required to establish the correct team, not used as a model input.
    for index, path in enumerate(files):  # Read one shard at a time to bound memory use.
        selected = latest_rows(path, columns, ["match_id", "slot"])  # Extract the last state at each requested time.
        player_parts.append(selected)  # Keep only the small extracted table.
        if index % 100 == 0:  # Report useful progress without printing every row.
            print(f"Player snapshots: {index + 1}/{len(files)} shards", flush=True)  # Make long scans visible.
    players = pd.concat(player_parts, ignore_index=True)  # Join the compact snapshots from every shard.
    keys = ["match_id", "horizon_seconds"]  # A match at one time is one modelling observation.
    assert not players.duplicated(keys + ["slot"]).any()  # Repeated player states would inflate team totals.
    slots = players.groupby(keys).slot.agg(["count", "nunique", "min", "max"])  # Check all ten player slots are observed.
    valid = slots.loc[(slots["count"] == 10) & (slots["nunique"] == 10) & (slots["min"] == 0) & (slots["max"] == 9)].index  # Exclude incomplete or malformed player lineups.
    players = players.set_index(keys).loc[valid].reset_index()  # Keep only complete observed matches at each time.
    constants = json.loads((ROOT / "data/raw/dota_constants/heroes.json").read_text())  # Use an attributed hero identifier dictionary, not guesses about slot order.
    normalise = lambda name: re.sub(r"[^a-z0-9]", "", str(name).lower().replace("npc_dota_hero_", ""))  # Make internal and readable names comparable without changing their meaning.
    aliases = {normalise(hero[field]): int(hero["id"]) for hero in constants.values() for field in ["name", "localized_name"]}  # Accept the dictionary's verified internal and display names.
    players["hero_id"] = players.hero.map(lambda name: aliases.get(normalise(name)))  # Unknown or blank identities remain missing rather than being assigned a team.
    team_maps = {int(row.match_id): {int(getattr(row, f"p{slot}_hero_id")): "radiant" if slot < 5 else "dire" for slot in range(10)} for row in metadata.itertuples()}  # Use the metadata draft to link each hero to its actual team.
    players["side"] = [team_maps[int(match_id)].get(int(hero_id)) if pd.notna(hero_id) else None for match_id, hero_id in zip(players.match_id, players.hero_id)]  # Parsed slot order is not assumed to encode teams.
    identity = players.groupby(keys).agg(heroes=("hero_id", "nunique"), labelled=("side", "count"))  # Require all ten distinct draft heroes to be identified.
    side_counts = players.groupby(keys + ["side"]).size().unstack("side", fill_value=0)  # Check that five observed players belong to each team.
    identity_valid = identity.loc[(identity.heroes == 10) & (identity.labelled == 10) & (side_counts.get("radiant", 0) == 5) & (side_counts.get("dire", 0) == 5)].index  # Exclude incomplete identity records instead of inventing a mapping.
    save_json("D_identity_audit.json", {"version": "metadata_draft_identity_v3", "snapshot_groups_before": len(identity), "snapshot_groups_valid": len(identity_valid), "unknown_name_counts": players.loc[players.hero_id.isna(), "hero"].fillna("").value_counts().to_dict(), "reason": "Observed replay slot order does not consistently match metadata team order; map verified hero identities to the metadata draft", "excluded": "Groups without ten distinct identified draft heroes and five per side"})  # Preserve this material correction as generated evidence.
    players = players.set_index(keys).loc[identity_valid].reset_index()  # Keep only snapshots with an auditable complete team mapping.
    totals = players.groupby(keys + ["side"])[STATS].sum()  # Summarise each team's currently observed statistics.
    radiant = totals.xs("radiant", level="side")  # Select one team using its side label.
    dire = totals.xs("dire", level="side")  # Select its opponent using the same match and time keys.
    differences = (radiant - dire).add_suffix("_diff")  # A positive number means a larger Radiant total.
    building_parts = []  # Prepare objective survival measures independently of player data.
    building_files = sorted((RAW / "buildings").glob("*.parquet"))  # Structures are another source table.
    for index, path in enumerate(building_files):  # Scan sequentially to avoid huge in memory tables.
        selected = latest_rows(path, ["match_id", "game_time", "entity_name", "entity_type", "team", "alive"], ["match_id", "entity_name"])  # Use only observed structure states.
        building_parts.append(selected)  # Store one state per building and time.
        if index % 100 == 0:  # Report the expensive second scan periodically.
            print(f"Building snapshots: {index + 1}/{len(building_files)} shards", flush=True)  # Show progress to the notebook reader.
    buildings = pd.concat(building_parts, ignore_index=True)  # Join the compact structure snapshots.
    for kind in ["tower", "barracks"]:  # These structures represent objective survival without using the winning ancient.
        subset = buildings.loc[buildings.entity_type == kind]  # Keep one interpretable structure family.
        if subset.empty:  # The downloaded source may omit a structure type described in its card.
            continue  # Exclude an unavailable feature rather than inventing zero observations.
        counts = subset.groupby(keys + ["team"]).alive.sum().unstack("team")  # Count alive structures on each side.
        differences[kind + "s_alive_diff"] = counts[2] - counts[3]  # Team codes two and three are Radiant and Dire in these state tables.
    frame = differences.reset_index()  # Return match and time keys to ordinary columns.
    frame = frame.merge(metadata[["match_id", "radiant_win", "start_time", "duration_sec"]], on="match_id", validate="many_to_one")  # Attach targets and an eligibility check separately from model inputs.
    frame = frame.loc[frame.horizon_seconds < frame.duration_sec].drop(columns="duration_sec")  # Include only matches still active at the observation time; duration never becomes a predictor.
    frame = frame.merge(partitions[["match_id", "partition"]], on="match_id", validate="many_to_one")  # Carry the same frozen assignment to every snapshot.
    features = [c for c in frame if c.endswith("_diff")]  # Downstream models can only use the approved difference columns.
    before = len(frame)  # Preserve the denominator before excluding incomplete objective snapshots.
    frame = frame.dropna(subset=features)  # Do not silently impute unverified structure coverage.
    assert np.isfinite(frame[features].to_numpy()).all()  # Reject invalid numeric inputs before fitting.
    frame.to_parquet(snapshot_path, index=False)  # Save the reusable derived data, excluded from Git by policy.
    save_json("D_preparation.json", {"features": features, "feature_units": "team totals except alive structure counts; Radiant minus Dire", "rows_before_complete_feature_filter": before, "rows_after": len(frame), "rows_by_horizon_partition": frame.groupby(["horizon_seconds", "partition"]).size().rename("matches").reset_index().to_dict("records"), "max_snapshot_age_seconds": 1.2, "target": "radiant_win", "source_tick_shards": len(files), "source_building_shards": len(building_files), "notes": ["No future values or final statistics used as inputs", "Each match at one horizon has ten observed player slots", "Later horizons describe matches still observed then", "No ward or item build claims are made"]})  # Save all coverage counts as generated evidence.
    print(f"Prepared {len(frame)} complete match snapshots", flush=True)  # Report the actual usable size.
    return frame  # Return the compact table for notebooks or command line use.


if __name__ == "__main__":  # Running the script directly prepares data but never trains a model.
    prepare()  # Execute the documented preparation workflow.
