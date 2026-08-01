#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""annAudit: single-file auditor for AnnData objects.

The auditor is read-only: it never changes ``adata``.

Scope
-----
The checks are intentionally conservative. They detect structural inconsistencies,
likely matrix-state mistakes, stale analysis metadata, graph problems, and common
spatial-coordinate/image mismatches. Matrix-state inference is heuristic and is
reported with evidence rather than treated as ground truth.

Dependencies
------------
Required: numpy, pandas, scipy
Runtime object: an AnnData-like object (normally ``anndata.AnnData``)
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

from .report import AuditReport, _Collector, __version__

from .utils import _check_object_state, _check_axis_tables, _check_axis_names, _profile_matrix, _report_matrix_profile, \
    _check_zero_axes, _check_layers, _check_raw, _check_var_annotations, _check_obs_annotations, \
    _check_multidimensional_annotations, _check_pairwise_matrices, _check_neighbors_metadata, _check_analysis_metadata, \
    _check_spatial, _check_color_metadata


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def audit_adata(
    adata: Any,
    *,
    max_matrix_sample: int = 200_000,
    max_rows_for_axis_checks: int = 4_000,
    random_state: int = 0,
    include_pass: bool = True,
) -> AuditReport:
    """Audit one AnnData object without modifying it.

    Parameters
    ----------
    adata
        An ``anndata.AnnData`` object or an AnnData-compatible object.
    max_matrix_sample
        Maximum sampled values per matrix for distribution/state checks.
    max_rows_for_axis_checks
        Maximum rows sampled for expensive per-axis checks on dense/backed data.
    random_state
        Seed used only for deterministic audit sampling.
    include_pass
        Keep successful checks in the returned report. Errors/warnings/info are
        always retained.

    Returns
    -------
    AuditReport
        Machine-readable findings plus text/JSON rendering methods.
    """
    if adata is None:
        raise TypeError("adata must be an AnnData object, not None")

    required_attrs = (
        "X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns",
        "n_obs", "n_vars", "obs_names", "var_names",
    )
    missing = [name for name in required_attrs if not hasattr(adata, name)]
    if missing:
        raise TypeError(
            "Object is not AnnData-compatible; missing attributes: " + ", ".join(missing)
        )

    rng = np.random.default_rng(random_state)
    c = _Collector()

    metadata: Dict[str, Any] = {
        "tool": "annAudit",
        "tool_version": __version__,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "object_type": f"{type(adata).__module__}.{type(adata).__name__}",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "is_view": bool(getattr(adata, "is_view", False)),
        "is_backed": bool(getattr(adata, "isbacked", False)),
        "filename": str(getattr(adata, "filename", "") or ""),
        "random_state": int(random_state),
        "sampling": {
            "max_matrix_sample": int(max_matrix_sample),
            "max_rows_for_axis_checks": int(max_rows_for_axis_checks),
        },
    }

    _check_object_state(adata, c)
    _check_axis_tables(adata, c)
    _check_axis_names(adata, c)

    x_profile = _profile_matrix(
        adata.X,
        rng=rng,
        max_values=max_matrix_sample,
        max_rows=max_rows_for_axis_checks,
        expected_shape=(adata.n_obs, adata.n_vars),
        name="X",
    )
    metadata["matrix_state"] = x_profile.get("state_inference", {})
    metadata["x_profile"] = x_profile
    _report_matrix_profile("X", x_profile, c, category="matrix")
    _check_zero_axes(
        adata.X,
        adata.n_obs,
        adata.n_vars,
        c,
        rng,
        max_rows_for_axis_checks,
    )

    _check_layers(
        adata,
        c,
        rng,
        max_matrix_sample=max_matrix_sample,
        max_rows=max_rows_for_axis_checks,
    )
    _check_raw(
        adata,
        c,
        rng,
        max_matrix_sample=max_matrix_sample,
        max_rows=max_rows_for_axis_checks,
    )
    _check_var_annotations(adata, c)
    _check_obs_annotations(adata, c)
    _check_multidimensional_annotations(adata, c, rng)
    _check_pairwise_matrices(adata, c)
    _check_neighbors_metadata(adata, c)
    _check_analysis_metadata(adata, x_profile, c)
    _check_spatial(adata, c)
    _check_color_metadata(adata, c)

    findings = c.findings if include_pass else [f for f in c.findings if f.severity != "PASS"]
    return AuditReport(metadata=metadata, findings=findings)
