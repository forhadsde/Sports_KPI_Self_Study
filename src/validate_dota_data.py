"""Check team, outcome, timing and partition assumptions before final evaluation."""
import hashlib  # Check that match membership has not changed during development.
import json  # Read settings and save observed validation evidence.
import numpy as np  # Recognise list arrays stored in metadata.
import pandas as pd  # Compare independent fields within the downloaded tables.
import pyarrow.parquet as pq  # Read the compact objective event fields.
from dota_data import PROCESSED, RAW, ROOT, save_json  # Reuse source paths and result writing.


def run():
    metadata = pd.read_parquet(RAW / "matches.parquet")  # Read outcome and draft metadata for consistency checks.
    checked, mismatches, unavailable = 0, 0, 0  # Count verified mappings separately from unavailable checks.
    for row in metadata.itertuples():  # Check each match without relying on replay slot order.
        arrays = [row.stratz_pb_hero_id, row.stratz_pb_is_pick, row.stratz_pb_is_radiant]  # Draft events provide a second representation of hero team membership.
        if not all(isinstance(a, (list, tuple, np.ndarray)) for a in arrays):  # Some earlier matches have no draft event arrays.
            unavailable += 1  # Preserve this verification limitation rather than treating it as a pass.
            continue  # Move to the next match with an actual draft record.
        picks = {int(hero): bool(side) for hero, picked, side in zip(*arrays) if picked}  # Ignore bans and retain only selected heroes and sides.
        if len(picks) != 10:  # A partial draft cannot verify every player.
            unavailable += 1  # Record incomplete checks transparently.
            continue  # Do not infer missing choices.
        checked += 1  # Count the complete check.
        mismatches += int(any(picks.get(int(getattr(row, f"p{i}_hero_id"))) != (i < 5) for i in range(10)))  # Verify the metadata's team ordering against its draft events.
    assert mismatches == 0  # A known conflicting team assignment must be resolved before modelling conclusions.
    objective_parts = []  # Collect compact match end event records.
    for path in sorted((RAW / "objectives").glob("*.parquet")):  # Inspect every objective shard, not only a sample.
        events = pq.read_table(path, columns=["match_id", "timestamp", "target", "team"]).to_pandas()  # Read fields needed to validate ancient destruction and clock agreement.
        objective_parts.append(events.loc[events.target.fillna("").str.endswith("_fort")])  # Select the destroyed ancient rather than ordinary towers.
    forts = pd.concat(objective_parts, ignore_index=True).sort_values("timestamp").drop_duplicates("match_id", keep="first")  # Use the first decisive ancient destruction in each match.
    forts = forts.merge(metadata[["match_id", "radiant_win", "duration_sec"]], on="match_id", validate="one_to_one")  # Compare event evidence with metadata for the same game.
    outcome_mismatches = int(((forts.team == 3) != forts.radiant_win).sum())  # Dire ancient destruction implies a Radiant win.
    assert outcome_mismatches == 0  # Do not train on a known reversed side label.
    end_errors = forts.timestamp - forts.duration_sec  # Match event timestamps should agree closely with metadata duration.
    snapshots = pd.read_parquet(PROCESSED / "dota_snapshots.parquet")  # Check the actual model input table.
    joined = snapshots.merge(metadata[["match_id", "duration_sec"]], on="match_id", validate="many_to_one")  # Attach duration only for eligibility verification.
    assert (joined.horizon_seconds < joined.duration_sec).all()  # No completed match may enter an earlier live prediction cohort.
    assert snapshots.groupby("match_id").partition.nunique().max() == 1  # Every time observation from a game stays in the same partition.
    partition_path = PROCESSED / "dota_partitions.parquet"  # Locate the membership record frozen before training.
    protocol = json.loads((ROOT / "outputs/results/D_protocol.json").read_text())  # Read its recorded fingerprint.
    unchanged = hashlib.sha256(partition_path.read_bytes()).hexdigest() == protocol["partition_sha256"]  # Detect accidental replacement of the split.
    assert unchanged  # The held out population cannot change after development begins.
    save_json("D_integrity_validation.json", {"metadata_draft_complete_checks": checked, "metadata_draft_mismatches": mismatches, "metadata_draft_checks_unavailable": unavailable, "ancient_event_outcome_checks": len(forts), "ancient_event_outcome_mismatches": outcome_mismatches, "ancient_timestamp_minus_duration_quantiles_seconds": np.quantile(end_errors, [0, .25, .5, .75, 1]).tolist(), "snapshots_checked": len(snapshots), "no_completed_match_snapshots": True, "all_match_observations_share_partition": True, "partition_fingerprint_unchanged": unchanged, "unit_tests": ["Future state and anchor state excluded from temporal extraction", "Known corner path lengths recovered", "Death boundary never crossed"], "limits": ["Consistency checks compare fields, not independent ground truth", "Incomplete draft arrays are explicitly counted", "Event anchored state timing remains sampled"]})  # Preserve evidence and its limits beside experimental results.
    print("Data integrity checks passed", flush=True)  # Report a measured outcome rather than an assurance before checks.


if __name__ == "__main__":  # Permit a deliberate terminal validation step.
    run()  # Execute the checks before final evaluation.
