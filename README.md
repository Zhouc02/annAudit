# annAudit

[![PyPI version](https://img.shields.io/pypi/v/annAudit.svg)](https://pypi.org/project/annAudit/) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/annaudit?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/annaudit)

**annAudit** is a read-only auditor for [`AnnData`](https://anndata.readthedocs.io/) objects used in single-cell and spatial omics workflows.

An `.h5ad` file can be readable and structurally valid while still containing silent inconsistencies: a normalized matrix stored as counts, duplicated identifiers, stale graph metadata, mismatched PCA components, or spatial coordinates that do not agree with the associated tissue image. annAudit inspects these object-level relationships and produces human-readable and machine-readable findings without modifying the input object.

> [!WARNING]
> annAudit is an auditing tool, not a cell-filtering pipeline, a schema validator, or a substitute for biological review.

## Features

annAudit currently checks common issues involving:

* the shape, type, finite values, sparsity, and likely state of `adata.X`;
* consistency of `adata.layers`, `adata.raw`, `adata.obs`, and `adata.var`;
* duplicated, empty, or inconsistent observation and variable identifiers;
* suspicious gene identifier patterns, including Ensembl version suffixes;
* all-zero observations or variables;
* categorical metadata, missing values, and clustering annotations;
* dimensions and finite values in `obsm`, `varm`, `obsp`, and `varp`;
* graph shape, symmetry, isolated nodes, and neighbor metadata;
* PCA scores, loadings, and stored analysis metadata;
* common Visium/Scanpy-style spatial coordinates, images, and scale factors;
* color metadata associated with categorical annotations.

The matrix-state assessment is conservative and heuristic. It reports the observed evidence rather than treating the inferred state as ground truth.

## Quick start

```Python
pip install annAudit
import annAudit as anndit
report = anndit.audit_adata(adata)
report.print()
```

annAudit does not modify `adata`.

## Python API

```python
report = anndit.audit_adata(
    adata,
    max_matrix_sample=200_000,
    max_rows_for_axis_checks=4_000,
    random_state=0,
    include_pass=True,
)
```

### Parameters

#### `adata`

An `anndata.AnnData` object or an AnnData-compatible object exposing the standard matrix and annotation attributes.

#### `max_matrix_sample`

Maximum number of matrix values sampled for distribution and matrix-state inspection.

Default:

```python
200_000
```

#### `max_rows_for_axis_checks`

Maximum number of rows sampled by potentially expensive per-axis checks on large dense or backed matrices.

Default:

```python
4_000
```

#### `random_state`

Seed used for deterministic audit sampling. It does not affect the input object.

Default:

```python
0
```

#### `include_pass`

Whether successful checks are retained in the returned report. Errors, warnings, and informational findings are always retained.

Default:

```python
True
```

To return only non-PASS findings:

```python
report = anndit.audit_adata(adata, include_pass=False)
```

## Severity levels

| Severity | Interpretation                                                                       |
| -------- | ------------------------------------------------------------------------------------ |
| `ERROR`  | A strong structural inconsistency or a condition likely to invalidate downstream use |
| `WARN`   | A plausible semantic problem that should be reviewed                                 |
| `INFO`   | Context, uncertainty, or a non-critical condition worth recording                    |
| `PASS`   | The corresponding check completed without detecting the targeted issue               |

A warning is not necessarily proof that the data are incorrect. In particular, matrix-state classification and some spatial checks rely on conservative heuristics.

## Example report

```text
============================================================
annAudit: AnnData audit, Version: 1.0.0
MIT License, Aug 1 2026, 22:25, UTC+8
============================================================
Shape: 3,484 observations × 18,085 variables
Findings: 1 ERROR, 1 WARN, 4 INFO, 7 PASS
X heuristic state: count_like (confidence=high)
-
[ERROR] NAME_VAR_DUP | var | Duplicate var_names
  Found 3 duplicated entries.
  Recommendation: Make var_names unique while retaining a stable mapping to original identifiers.
  Evidence: {"examples": ["TBCE", "HSPA14", "TMSB15B"]}

[WARN] MAT_ZERO_VAR | matrix | All-zero variables detected
  Found 31 all-zero variables (0.17%).
  Recommendation: Remove uninformative variables unless they are deliberately retained for panel completeness.

[INFO] LAYER_NONE | layers | No layers present
  The object stores no alternate expression matrices in .layers.
  Recommendation: For reproducibility, consider retaining raw counts in layers['counts'] when appropriate.

[INFO] MAT_X_STATE | matrix | Heuristic state for matrix 'X'
  Inferred 'count_like' with high confidence. This is a heuristic, not proof of preprocessing history.
  Recommendation: Compare this inference with the recorded workflow and source files.
  Evidence: {"dtype": "float32", "integer_like_fraction": 1.0, "max": 1031.0, "min": 1.0, "negative_fraction": 0.0, "q99": 6.0, "reason": "nonnegative and almost all sampled values are integer-like", "sample_size": 200000, "sampling_mode": "sparse_nonzero_sample", "zero_fraction_estimate": 0.8916888198889857}

[INFO] RAW_NONE | raw | No .raw snapshot present
  The object does not contain an AnnData.raw snapshot.
  Recommendation: This is valid if raw counts or preprocessed values are stored and documented elsewhere.

[INFO] SPAT_UNS_MISSING | spatial | Coordinates lack .uns['spatial'] image metadata
  Spatial coordinates are present, but no Scanpy/Visium-style image metadata was found.
  Recommendation: This is valid for platforms without tissue images; otherwise retain images, scale factors, and units.
```

The exact findings depend on the contents of the audited object.

## Matrix sampling behavior

annAudit avoids loading a large matrix in full whenever possible.

For sparse matrices:

* stored non-zero values are sampled for distribution and state inference;
* the implicit-zero fraction is recorded separately;
* reported sparse distribution statistics therefore describe the sampled non-zero values, not the complete matrix including implicit zeros.

For dense or backed matrices:

* a bounded subset of entries is sampled;
* the sampling seed is recorded in the report metadata;
* a sampled check cannot prove that a rare issue is absent from the complete object.

Use the same `random_state` when comparing repeated audit runs.

## What annAudit does not do

annAudit does **not**:

* remove low-quality cells or spots;
* normalize expression values;
* select highly variable genes;
* rerun PCA, neighborhood construction, or clustering;
* automatically rotate, flip, or rescale spatial coordinates;
* silently repair high-risk inconsistencies;
* determine the biological correctness of a cell-type or spatial-domain label;
* guarantee that a sampled audit detects every rare anomaly;
* replace submission-specific schema validators.

The recommended workflow is:

1. audit the object;
2. review the evidence and recommendations;
3. repair the object in the original analysis pipeline;
4. regenerate derived representations when necessary;
5. run annAudit again on the reconstructed object.

## Read-only design

`audit_adata()` is designed not to mutate the supplied object. It returns an
`AuditReport` containing metadata and findings.

A typical workflow is:

```python
before_shape = adata.shape
report = anndit.audit_adata(adata)
assert adata.shape == before_shape
```

Potential repairs should be performed explicitly by the user so that changes remain traceable and reproducible.
