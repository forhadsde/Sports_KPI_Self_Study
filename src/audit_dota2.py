"""Record dataset metadata and limited schema checks without training models."""
from collections import defaultdict  # Create an empty counter record automatically for each new table family.
import json  # Read acquisition evidence and write inspectable audit measurements.
from pathlib import Path  # Resolve file locations independently of the terminal folder.

import pandas as pd  # Summarise metadata and selected source rows as labelled tables.
import pyarrow.parquet as pq  # Read file metadata without loading billions of observations into memory.

ROOT = Path(__file__).resolve().parents[1]  # Locate the project from this script.
DATA = ROOT / "data/raw/betty-dota2"  # Inspect the untouched downloads.
OUT = ROOT / "outputs/results/D_data_audit.json"  # Give the audit a stable evidence filename.


def main():
    manifest = json.loads((OUT.parent / "D_download_manifest.json").read_text(encoding="utf-8"))  # Link schema evidence to the exact acquired revision.
    result = dict(download_status=manifest["status"], revision=manifest["revision"],
                  audit_date="2026-10-29", tables={}, samples={})
    totals = defaultdict(lambda: dict(files=0, rows=0, bytes=0, schemas=[]))  # Start a separate inventory for each table type.
    for path in sorted(DATA.rglob("*.parquet")):  # Inventory every Parquet file in a reproducible order.
        family = path.parent.name if path.parent != DATA else "matches"  # Distinguish the root metadata table from replay table folders.
        meta = pq.ParquetFile(path)  # Open the compact header rather than all source rows.
        row = totals[family]  # Retrieve this family's running counters.
        row["files"] += 1  # Count the current source shard once.
        row["rows"] += meta.metadata.num_rows  # Add its observed row count from the header.
        row["bytes"] += path.stat().st_size  # Record actual disk size rather than a rounded estimate.
        cols = meta.schema_arrow.names  # List the fields the file actually advertises.
        if cols not in row["schemas"]:  # Detect column list changes between shards.
            row["schemas"].append(cols)  # Preserve each distinct observed column list.
    result["tables"] = dict(totals)  # Store the completed full inventory.
    matches = pd.read_parquet(DATA / "matches.parquet")  # Load the small complete match metadata population.
    result["matches"] = dict(rows=len(matches), unique_matches=int(matches.match_id.nunique()),
                             duplicate_ids=int(matches.match_id.duplicated().sum()),
                             columns=matches.columns.tolist(),
                             missing_by_column=matches.isna().sum().astype(int).to_dict(),
                             empty_string_by_column={c: int((matches[c] == "").sum()) for c in matches.select_dtypes(include="object") if not c.endswith("_per_min") and not c.startswith(("stratz_pb_", "replay_pb_"))},
                             first_match_utc=str(pd.to_datetime(matches.start_time.min(), unit="s", utc=True)),
                             last_match_utc=str(pd.to_datetime(matches.start_time.max(), unit="s", utc=True)),
                             status_counts=matches.status.value_counts(dropna=False).to_dict())
    for family in ["ticks", "wards", "objectives", "events", "actions", "buildings"]:  # Inspect detailed completeness only in the documented sample families.
        paths = sorted((DATA / family).glob("*.parquet"))  # Select the same first shard on repeated audits.
        if not paths:  # An absent table has no sample to inspect.
            continue  # Keep absence visible in the inventory rather than inventing observations.
        sample = pd.read_parquet(paths[0])  # Bound memory by loading one shard, not the whole population.
        row = dict(file=str(paths[0].relative_to(DATA)), rows=len(sample),
                   unique_matches=int(sample.match_id.nunique()),
                   missing_by_column=sample.isna().sum().astype(int).to_dict())
        for col in ["ward_event_type", "objective_type", "event_type", "team"]:  # Inspect category meanings before treating codes as measurements.
            if col in sample:  # Different families expose different fields.
                row[col + "_counts"] = {str(k): int(v) for k, v in sample[col].value_counts(dropna=False).items()}  # Retain missing categories rather than hiding them.
        if family == "ticks":  # Player states support additional sampling and inventory checks.
            row["duplicate_match_slot_time"] = int(sample.duplicated(["match_id", "slot", "game_time"]).sum())  # Check repeated observation keys within the sampled shard.
            steps = sample.sort_values(["match_id", "slot", "game_time"]).groupby(["match_id", "slot"]).game_time.diff().dropna()  # Measure within player replay intervals without crossing match boundaries.
            row["sampling_interval_seconds"] = {str(k): float(v) for k, v in steps.quantile([0, .5, .95, 1]).items()}  # Summarise typical and extreme replay intervals.
            row["generic_item_counts"] = {c: int((sample[c] == "CDOTA_Item").sum()) for c in sample if c.startswith("item_")}  # Detect placeholder item names that cannot support detailed builds.
        result["samples"][family] = row  # Label sample findings separately from full inventory counts.
    result["limitations"] = [
        "Detailed null and sampling checks cover the first shard of each listed family only, not every row.",
        "Ward and objective coordinates can be absent; do not assume spatial vision coverage can be reconstructed.",
        "Generic item strings occur; detailed item interpretation requires further validation.",
        "Team semantics differ between event types and must be checked against source code.",
        "This acquisition audit does not fit models or evaluate predictions; later study stages are recorded separately.",
    ]
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")  # Save measured evidence and its limitations together.
    print(json.dumps({"matches": result["matches"]["rows"], "tables": {k: {n: v[n] for n in ["files", "rows"]} for k, v in totals.items()}, "download_status": result["download_status"]}, indent=2))


if __name__ == "__main__":  # An import must not trigger a source scan.
    main()  # Audit only when this script is deliberately executed.
