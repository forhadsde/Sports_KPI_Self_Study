"""Check delivery consistency using saved evidence, without fitting or prediction."""
from pathlib import Path  # Locate the project and its evidence.
import hashlib  # Preserve file identities in the verification record.
import json  # Read measurements and save the verification outcome.
import re  # Check citation keys against the bibliography.
import nbformat  # Inspect notebook execution status without executing cells.
import pandas as pd  # Compare per seed measurements with the public summary.
import numpy as np  # Compare floating point values with a numerical tolerance.

ROOT = Path(__file__).resolve().parents[1]  # Find the shared project folder.
RESULTS = ROOT / "outputs/results"  # Keep checks next to the measured results.
def load(name):  # Use one evidence reader for consistent encoding.
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))  # Decode labelled saved measurements.

summary = load("D_summary.json")  # Load the source of public result tables.
final = pd.DataFrame(load("D_final_metrics.json")["records"])  # Read original per seed metrics.
for row in summary["test_table"]:  # Verify every table condition rather than only the best model.
    group = final.loc[(final.horizon_seconds == row["horizon_seconds"]) & (final.model == row["model"]) & (final.probability == row["probability"])]  # Match the exact experimental identity.
    assert len(group) == len(summary["protocol"]["seeds"])  # Require every planned seed.
    for metric in ["auc", "accuracy", "brier", "log_loss"]:  # Check discrimination and probability quality.
        assert np.isclose(group[metric].mean(), row[metric + "_mean"])  # Detect a manually altered summary value.
        assert np.isclose(group[metric].std(ddof=1), row[metric + "_sd"])  # Preserve the stated seed spread definition.
    assert group.matches.nunique() == 1 and group.matches.iloc[0] == row["matches"]  # Require identical eligible populations within each comparison.
manifest = load("D_download_manifest.json")  # Read acquisition completion and verification records.
assert manifest["status"] == "complete" and not manifest["failures"]  # Do not deliver an incomplete acquisition as finished.
assert len(manifest["files"]) == manifest["expected_files"]  # Check inventory completeness.
assert manifest["verified_bytes"] == manifest["expected_bytes"]  # Check aggregate byte agreement.
protocol = load("D_protocol.json")  # Preserve the completed final evaluation boundary.
assert protocol["analysis_frozen"] and "complete" in protocol["status"]  # Do not mislabel a development run as final.
notebooks = []  # Record the actual executed code cell counts.
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):  # Check every active notebook.
    book = nbformat.read(path, as_version=4)  # Read cells and outputs without running them.
    cells = [cell for cell in book.cells if cell.cell_type == "code"]  # Markdown cells need no execution count.
    assert all(cell.execution_count is not None for cell in cells)  # Require top to bottom execution evidence.
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)  # Reject a notebook containing a failed cell.
    notebooks.append(dict(filename=path.name, code_cells=len(cells), errors=0))  # Save the checked facts.
report_checks = {}  # The private report is optional in a public clone.
if (ROOT / "report/main.tex").exists():  # Check the locally reviewed report when available.
    tex = (ROOT / "report/main.tex").read_text(encoding="utf-8")  # Inspect actual citation commands.
    keys = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}  # Expand multiple keys in one citation.
    bibliography = (ROOT / "report/references.bib").read_text(encoding="utf-8")  # Read the verified bibliography.
    known = set(re.findall(r"@\w+\{([^,]+)", bibliography))  # Identify defined reference keys.
    assert keys and not (keys - known)  # Reject absent references and a broken citation parser.
    report_checks = dict(cited_sources=len(keys), bibliography_entries=len(known), missing_citation_keys=[])  # Record bibliographic consistency, not a claim of new source review.
record = dict(status="passed", summary_conditions_checked=len(summary["test_table"]), per_seed_records_checked=len(final), notebooks=notebooks, report=report_checks, final_metrics_sha256=hashlib.sha256((RESULTS / "D_final_metrics.json").read_bytes()).hexdigest(), final_prediction_sha256=hashlib.sha256((ROOT / "outputs/predictions/D_final_predictions.parquet").read_bytes()).hexdigest(), scope="Saved evidence consistency, notebook execution and citation identity checks; no fitting or final model reevaluation")  # Distinguish delivery checks from new scientific results.
(RESULTS / "D_delivery_validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")  # Preserve the verification evidence.
print(json.dumps(record, indent=2))  # Make the completed checks reviewable.
