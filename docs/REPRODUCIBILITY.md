# Reproducibility notes

The executed notebooks are teaching documents for the completed experiment. They read saved measurements and display the commented shared implementation. They do not automatically fit new models or evaluate the final period again. The data audit notebook reads the saved acquisition audit unless it is absent. Source rescanning is a deliberate terminal operation.

The README gives the scientific run sequence for a separate fresh experiment. The published final results deliberately activate a guard that prevents accidental overwriting. Reproducing a completed experiment is different from using its final outcomes to choose a better approach. A new scientific extension requires a new evaluation period.

## Environment

`environment.yml` pins the principal packages to the observed environment. `outputs/results/D_environment_versions.json` records the installed scientific and notebook package versions. `D_runtime.json` additionally records the GPU, CUDA build and computation check. Platform specific libraries and floating point arithmetic may still affect exact numerical agreement.

## Frozen evidence

`D_protocol.json` preserves the historical partition fingerprint and implementation hashes recorded before final evaluation. These hashes are not replaced by current file hashes. Subsequent packaging changes added shared seed imports, copies of read only label buffers, comments, teaching notebooks and explicit reproduction commands. The seed values and fitted scientific choices were preserved. Downloader improvements pin the historical revision even when no earlier manifest exists.

The final models were not refitted after opening the final period. Summary tables, paired intervals and plots use saved predictions. A mean of seed metrics differs mathematically from a metric computed after averaging seed probabilities; both are labelled explicitly.

## Timing metadata

The legacy `max_snapshot_age_seconds` field in `D_preparation.json` describes the state matching component alone. It is not the combined match clock uncertainty. The event anchor tolerance and preceding state tolerance give the conservative combined bound recorded as `snapshot_age_bound_seconds` in `D_protocol.json`. The notebooks and report explain this distinction.

## Provenance and verification

The acquisition manifest records pinned source files and hashes. The integrity record distinguishes checks completed across the population from field inspection on sampled shards. Successful internal consistency checks do not establish independent ground truth.

The literature tracker preserves source identities, URLs, short summaries, access levels and search records. A checked bibliographic identity is not proof of every experimental detail. The report cites only the sources needed for its claims. The full review includes additional contextual sources that are not all cited in the short report.
