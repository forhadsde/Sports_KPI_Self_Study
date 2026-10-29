"""Record dataset metadata and limited schema checks without training models."""
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/betty-dota2"
OUT = ROOT / "outputs/results/D_data_audit.json"


def main():
    manifest = json.loads((OUT.parent / "D_download_manifest.json").read_text(encoding="utf-8"))
    result = dict(download_status=manifest["status"], revision=manifest["revision"],
                  audit_date="2026-10-29", tables={}, samples={})
    totals = defaultdict(lambda: dict(files=0, rows=0, bytes=0, schemas=[]))
    for path in sorted(DATA.rglob("*.parquet")):
        family = path.parent.name if path.parent != DATA else "matches"
        meta = pq.ParquetFile(path)
        row = totals[family]
        row["files"] += 1
        row["rows"] += meta.metadata.num_rows
        row["bytes"] += path.stat().st_size
        cols = meta.schema_arrow.names
        if cols not in row["schemas"]:
            row["schemas"].append(cols)
    result["tables"] = dict(totals)
    matches = pd.read_parquet(DATA / "matches.parquet")
    result["matches"] = dict(rows=len(matches), unique_matches=int(matches.match_id.nunique()),
                             duplicate_ids=int(matches.match_id.duplicated().sum()),
                             columns=matches.columns.tolist(),
                             missing_by_column=matches.isna().sum().astype(int).to_dict(),
                             empty_string_by_column={c: int((matches[c] == "").sum()) for c in matches.select_dtypes(include="object") if not c.endswith("_per_min") and not c.startswith(("stratz_pb_", "replay_pb_"))},
                             first_match_utc=str(pd.to_datetime(matches.start_time.min(), unit="s", utc=True)),
                             last_match_utc=str(pd.to_datetime(matches.start_time.max(), unit="s", utc=True)),
                             status_counts=matches.status.value_counts(dropna=False).to_dict())
    for family in ["ticks", "wards", "objectives", "events", "actions", "buildings"]:
        paths = sorted((DATA / family).glob("*.parquet"))
        if not paths:
            continue
        sample = pd.read_parquet(paths[0])
        row = dict(file=str(paths[0].relative_to(DATA)), rows=len(sample),
                   unique_matches=int(sample.match_id.nunique()),
                   missing_by_column=sample.isna().sum().astype(int).to_dict())
        for col in ["ward_event_type", "objective_type", "event_type", "team"]:
            if col in sample:
                row[col + "_counts"] = {str(k): int(v) for k, v in sample[col].value_counts(dropna=False).items()}
        if family == "ticks":
            row["duplicate_match_slot_time"] = int(sample.duplicated(["match_id", "slot", "game_time"]).sum())
            steps = sample.sort_values(["match_id", "slot", "game_time"]).groupby(["match_id", "slot"]).game_time.diff().dropna()
            row["sampling_interval_seconds"] = {str(k): float(v) for k, v in steps.quantile([0, .5, .95, 1]).items()}
            row["generic_item_counts"] = {c: int((sample[c] == "CDOTA_Item").sum()) for c in sample if c.startswith("item_")}
        result["samples"][family] = row
    result["limitations"] = [
        "Detailed null and sampling checks cover the first shard of each listed family only, not every row.",
        "Ward and objective coordinates can be absent; do not assume spatial vision coverage can be reconstructed.",
        "Generic item strings occur; detailed item interpretation requires further validation.",
        "Team semantics differ between event types and must be checked against source code.",
        "This acquisition audit does not fit models or evaluate predictions; later study stages are recorded separately.",
    ]
    OUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"matches": result["matches"]["rows"], "tables": {k: {n: v[n] for n in ["files", "rows"]} for k, v in totals.items()}, "download_status": result["download_status"]}, indent=2))


if __name__ == "__main__":
    main()
