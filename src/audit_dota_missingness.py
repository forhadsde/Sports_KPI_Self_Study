"""Additional post study source null audit, without fitting or test selection.

This check was added while explaining the methodology. It did not select the
features, change the frozen models or repeat final prediction evaluation.
Parquet null counts do not establish semantic correctness or detect every
possible floating point NaN stored as a non-null value.
"""
from datetime import datetime, timezone  # Record when this additional check ran.
from pathlib import Path  # Keep every read and result inside the project.
import json  # Save exact counts as inspectable evidence.
import pyarrow.parquet as pq  # Inspect source metadata without loading the full tables.

ROOT = Path(__file__).resolve().parents[1]  # Locate the study independently of the terminal folder.
FIELDS = ["net_worth", "xp", "kills", "assists", "last_hits", "denies", "level", "hp", "is_alive"]  # Audit the actual player statistics used in team totals.


def run():
    counts = {name: 0 for name in FIELDS}  # Start one null counter per source feature.
    unknown = {name: 0 for name in FIELDS}  # Count rows lacking available null metadata separately.
    rows = 0  # Record the full inspected source population.
    files = sorted((ROOT / "data/raw/betty-dota2/ticks").glob("*.parquet"))  # Include every acquired player state shard.
    assert files, "Download the pinned dataset before running this source audit."  # Do not report an empty scan as a successful census.
    for path in files:  # Bound memory by opening one source header at a time.
        metadata = pq.ParquetFile(path).metadata  # Read recorded row group statistics rather than billions of cells.
        rows += metadata.num_rows  # Accumulate the exact number of source rows covered.
        for i in range(metadata.num_row_groups):  # Inspect every row group in the current file.
            group = metadata.row_group(i)  # Obtain this group's per column statistics.
            for j in range(group.num_columns):  # Locate fields by their stored names rather than fixed column order.
                column = group.column(j)  # Inspect one recorded column chunk.
                name = column.path_in_schema  # Preserve the source field identity.
                if name not in counts:  # Exclude columns that were not model player statistics.
                    continue  # Keep the audit aligned with the predictor's actual inputs.
                statistics = column.statistics  # Null counts may be absent in some Parquet files.
                if statistics is None or statistics.null_count is None:  # An unavailable statistic is not zero missingness.
                    unknown[name] += group.num_rows  # Disclose the unchecked population for this field.
                else:  # Count only values supported by stored metadata.
                    counts[name] += statistics.null_count  # Preserve exact observed null counts.
    record = dict(stage="Additional post study validation for methodology explanation", checked_utc=datetime.now(timezone.utc).isoformat(), files=len(files), rows=rows, null_counts=counts, rows_without_null_statistics=unknown, limits=["Header null statistics do not independently establish value semantics", "Non-null floating point NaNs are not counted by this check", "The original preparation separately rejects non-finite aggregate feature rows", "This check did not change fitted models or repeat final evaluation"] )  # Keep the timing and limits next to the result.
    output = ROOT / "outputs/results/D_poststudy_missingness_audit.json"  # Save the additional evidence separately from the original acquisition audit.
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")  # Do not retrospectively replace the original audit.
    print(json.dumps(record, indent=2))  # Display the actual checked population and counts.
    return record  # Allow the methodology deck to cite the saved record.


if __name__ == "__main__":  # Importing documentation helpers must not trigger a source scan.
    run()  # Execute only when deliberately requested.
