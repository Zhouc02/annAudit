from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple
from .report import _Collector

import numpy as np
import pandas as pd
from scipy import sparse


# -----------------------------------------------------------------------------
# Object and axis checks
# -----------------------------------------------------------------------------


def _check_object_state(adata: Any, c: _Collector) -> None:
    if adata.n_obs <= 0 or adata.n_vars <= 0:
        c.add(
            "OBJ001", "ERROR", "object", "Empty AnnData axis",
            f"AnnData has shape ({adata.n_obs}, {adata.n_vars}).",
            "Restore a non-empty observation-by-variable matrix before analysis.",
        )
    else:
        c.add(
            "OBJ001", "PASS", "object", "Non-empty AnnData",
            f"AnnData has {adata.n_obs:,} observations and {adata.n_vars:,} variables.",
        )

    if getattr(adata, "is_view", False):
        c.add(
            "OBJ002", "WARN", "object", "AnnData is a view",
            "The object is a view of another AnnData object. Later writes may materialize a copy "
            "or behave differently from an independent object.",
            "Use adata = adata.copy() before editing or serializing the audited object.",
        )
    else:
        c.add("OBJ002", "PASS", "object", "AnnData is not a view", "The object is independent.")

    if getattr(adata, "isbacked", False):
        c.add(
            "OBJ003", "INFO", "object", "Backed mode detected",
            "The object is backed on disk. annAudit uses sampling for large matrices and does not "
            "force the full matrix into memory.",
        )


def _check_axis_tables(adata: Any, c: _Collector) -> None:
    for axis, table, expected in (
        ("obs", adata.obs, adata.n_obs),
        ("var", adata.var, adata.n_vars),
    ):
        if not isinstance(table, pd.DataFrame):
            c.add(
                f"AXIS_{axis.upper()}_TYPE", "ERROR", axis,
                f".{axis} is not a pandas DataFrame",
                f"Observed type: {type(table).__name__}.",
                f"Reconstruct .{axis} as a pandas DataFrame indexed to the corresponding AnnData axis.",
            )
            continue
        if len(table) != expected:
            c.add(
                f"AXIS_{axis.upper()}_LEN", "ERROR", axis,
                f".{axis} length does not match AnnData shape",
                f".{axis} has {len(table):,} rows but expected {expected:,}.",
                "Rebuild the object; axis metadata and matrix rows/columns are inconsistent.",
            )
        else:
            c.add(
                f"AXIS_{axis.upper()}_LEN", "PASS", axis,
                f".{axis} length matches AnnData shape",
                f".{axis} has the expected {expected:,} rows.",
            )

        duplicated_columns = table.columns[table.columns.duplicated()].tolist()
        if duplicated_columns:
            c.add(
                f"AXIS_{axis.upper()}_COL_DUP", "WARN", axis,
                f"Duplicate column names in .{axis}",
                f"Found {len(duplicated_columns)} duplicated metadata column names.",
                "Rename or merge duplicated columns before downstream selection by column name.",
                {"examples": duplicated_columns[:10]},
            )

        for col in table.columns:
            s = table[col]
            missing_fraction = float(s.isna().mean()) if len(s) else 0.0
            if missing_fraction == 1.0:
                c.add(
                    f"AXIS_{axis.upper()}_ALLNA_{_safe_id(col)}", "WARN", axis,
                    f"Column '{col}' is entirely missing",
                    f"All {len(s):,} values are missing.",
                    "Remove the column or repopulate it from source metadata.",
                )
            elif missing_fraction >= 0.5:
                c.add(
                    f"AXIS_{axis.upper()}_HIGHNA_{_safe_id(col)}", "INFO", axis,
                    f"Column '{col}' has high missingness",
                    f"Missing fraction is {missing_fraction:.1%}.",
                    "Confirm whether missingness is expected before using this field in stratified analyses.",
                )

            if isinstance(s.dtype, pd.CategoricalDtype):
                observed = int(s.nunique(dropna=True))
                declared = len(s.cat.categories)
                unused = declared - observed
                if unused > 0:
                    c.add(
                        f"AXIS_{axis.upper()}_UNUSED_CAT_{_safe_id(col)}", "INFO", axis,
                        f"Categorical column '{col}' contains unused categories",
                        f"{unused} of {declared} categories are unused.",
                        f"Optionally run adata.{axis}['{col}'] = adata.{axis}['{col}'].cat.remove_unused_categories().",
                    )


def _check_axis_names(adata: Any, c: _Collector) -> None:
    for axis, names in (("obs", pd.Index(adata.obs_names)), ("var", pd.Index(adata.var_names))):
        dup_count = int(names.duplicated().sum())
        if dup_count:
            examples = names[names.duplicated()].astype(str).unique()[:10].tolist()
            c.add(
                f"NAME_{axis.upper()}_DUP", "ERROR", axis,
                f"Duplicate {axis}_names",
                f"Found {dup_count:,} duplicated entries.",
                f"Make {axis}_names unique while retaining a stable mapping to original identifiers.",
                {"examples": examples},
            )
        else:
            c.add(
                f"NAME_{axis.upper()}_DUP", "PASS", axis,
                f"{axis}_names are unique",
                f"All {len(names):,} identifiers are unique.",
            )

        as_str = names.astype(str)
        empty_mask = np.array([x.strip() == "" or x.lower() in {"nan", "none"} for x in as_str])
        if int(empty_mask.sum()):
            c.add(
                f"NAME_{axis.upper()}_EMPTY", "ERROR", axis,
                f"Empty or placeholder {axis}_names",
                f"Found {int(empty_mask.sum()):,} empty/'nan'/'none' identifiers.",
                "Replace placeholder identifiers with stable unique IDs.",
            )

        if not all(isinstance(x, str) for x in names[: min(len(names), 1000)]):
            c.add(
                f"NAME_{axis.upper()}_NONSTR", "INFO", axis,
                f"Some {axis}_names are not strings",
                "Non-string index values can lead to inconsistent serialization or joins.",
                f"Convert using adata.{axis}_names = adata.{axis}_names.astype(str) after preserving originals.",
            )


# -----------------------------------------------------------------------------
# Matrix profiling
# -----------------------------------------------------------------------------


def _profile_matrix(
    matrix: Any,
    *,
    rng: np.random.Generator,
    max_values: int,
    max_rows: int,
    expected_shape: Tuple[int, int],
    name: str,
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "name": name,
        "type": f"{type(matrix).__module__}.{type(matrix).__name__}" if matrix is not None else "None",
        "expected_shape": list(expected_shape),
    }
    if matrix is None:
        profile.update({"present": False, "state_inference": {"state": "missing", "confidence": "high"}})
        return profile

    profile["present"] = True
    shape = tuple(int(x) for x in getattr(matrix, "shape", ()))
    profile["shape"] = list(shape)
    profile["shape_matches"] = shape == expected_shape
    profile["is_sparse"] = bool(sparse.issparse(matrix) or _is_sparse_dataset(matrix))
    profile["dtype"] = str(getattr(matrix, "dtype", "unknown"))

    try:
        values, zero_fraction_estimate, sampling_mode = _sample_matrix_values(
            matrix, rng=rng, max_values=max_values, max_rows=max_rows
        )
        profile["sampling_mode"] = sampling_mode
        profile["sample_size"] = int(values.size)
        profile["zero_fraction_estimate"] = _finite_or_none(zero_fraction_estimate)
    except Exception as exc:  # audit must continue even for unusual backed arrays
        profile["sampling_error"] = f"{type(exc).__name__}: {exc}"
        profile["state_inference"] = {"state": "unknown", "confidence": "low"}
        return profile

    if values.size == 0:
        profile.update(
            {
                "finite_fraction": None,
                "state_inference": {"state": "empty_or_all_zero", "confidence": "medium"},
            }
        )
        return profile

    numeric = np.asarray(values)
    if not np.issubdtype(numeric.dtype, np.number):
        profile["numeric"] = False
        profile["state_inference"] = {"state": "non_numeric", "confidence": "high"}
        return profile

    profile["numeric"] = True
    numeric = numeric.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    profile["finite_fraction"] = float(finite.mean())
    finite_values = numeric[finite]
    if finite_values.size == 0:
        profile["state_inference"] = {"state": "non_finite", "confidence": "high"}
        return profile

    profile.update(_numeric_summary(finite_values))
    profile["state_inference"] = _infer_matrix_state(finite_values)
    return profile


def _sample_matrix_values(
    matrix: Any,
    *,
    rng: np.random.Generator,
    max_values: int,
    max_rows: int,
) -> Tuple[np.ndarray, Optional[float], str]:
    """Sample matrix values without loading a large matrix in full.

    Notes
    -----
    For scipy/backed sparse matrices, returned ``values`` are sampled from
    *stored non-zero values*. The implicit-zero proportion is returned
    separately as ``zero_fraction``. This avoids allowing a highly sparse
    log-normalized matrix to look count-like merely because most entries are
    exact zeros.

    For dense matrices, returned values are sampled from a bounded rectangular
    set of matrix entries. At most ``max_values`` entries are materialized.
    """
    if max_values < 1:
        raise ValueError("max_values must be at least 1")
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")

    shape = tuple(int(x) for x in getattr(matrix, "shape", ()))
    if len(shape) != 2:
        raise ValueError(f"matrix must be two-dimensional; observed shape {shape}")
    n_rows, n_cols = shape
    total = n_rows * n_cols
    if total == 0:
        return np.array([], dtype=float), None, "empty"

    def take_sample(values: np.ndarray, size: int) -> np.ndarray:
        values = np.asarray(values).reshape(-1)
        if values.size <= size:
            return values.copy() if not values.flags.writeable else values
        idx = rng.choice(values.size, size=size, replace=False)
        return values[idx]

    def merge_reservoir(
        reservoir: Optional[np.ndarray],
        seen: int,
        chunk: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Merge one chunk into an exact uniform reservoir of non-zero values."""
        chunk = np.asarray(chunk).reshape(-1)
        if chunk.size:
            chunk = chunk[chunk != 0]
        m = int(chunk.size)
        if m == 0:
            if reservoir is None:
                reservoir = np.array([], dtype=float)
            return reservoir, seen

        new_seen = seen + m
        target = min(max_values, new_seen)
        if seen == 0 or reservoir is None or reservoir.size == 0:
            return take_sample(chunk, target), new_seen

        # Among a uniform sample of ``target`` from all old+new values, the
        # number contributed by the new chunk follows a hypergeometric law.
        take_new = int(rng.hypergeometric(ngood=m, nbad=seen, nsample=target))
        keep_old = target - take_new

        if keep_old < reservoir.size:
            old_idx = rng.choice(reservoir.size, size=keep_old, replace=False)
            old_part = reservoir[old_idx]
        else:
            old_part = reservoir

        new_part = take_sample(chunk, take_new) if take_new else chunk[:0]
        if old_part.size == 0:
            merged = new_part
        elif new_part.size == 0:
            merged = old_part
        else:
            merged = np.concatenate([old_part, new_part])
        return merged, new_seen

    if sparse.issparse(matrix):
        data = np.asarray(matrix.data).reshape(-1)  # BSR/DIA data may be multidimensional.
        nonzero_mask = data != 0
        stored_nonzero = int(np.count_nonzero(nonzero_mask))

        # Exact for canonical CSR/CSC and ordinary sparse matrices without
        # duplicate coordinates. Non-canonical duplicate coordinates can make
        # this an approximation, so the mode explicitly records that case.
        canonical = bool(getattr(matrix, "has_canonical_format", True))
        logical_nonzero_estimate = min(total, stored_nonzero)
        zero_fraction = 1.0 - logical_nonzero_estimate / float(total)
        zero_fraction = float(np.clip(zero_fraction, 0.0, 1.0))

        if stored_nonzero == 0:
            return np.array([], dtype=data.dtype), 1.0, "sparse_all_zero"

        nonzero_values = data if stored_nonzero == data.size else data[nonzero_mask]
        values = take_sample(nonzero_values, max_values)
        mode = "sparse_nonzero_sample"
        if not canonical:
            mode += "_noncanonical_zero_fraction_estimate"
        return values, zero_fraction, mode

    # Backed sparse datasets: sample rows, estimate sparsity from those rows,
    # and keep a bounded reservoir instead of concatenating every stored value.
    if _is_sparse_dataset(matrix):
        row_idx = np.sort(_sample_indices(n_rows, min(max_rows, n_rows), rng))
        reservoir: Optional[np.ndarray] = None
        seen_nonzero = 0
        sampled_nonzero = 0
        sampled_total = 0

        for start in range(0, len(row_idx), 128):
            rows = row_idx[start : start + 128]
            block = matrix[rows, :]
            if hasattr(block, "to_memory"):
                block = block.to_memory()

            if sparse.issparse(block):
                raw = np.asarray(block.data).reshape(-1)
                sampled_nonzero += int(np.count_nonzero(raw))
                sampled_total += int(block.shape[0] * block.shape[1])
                reservoir, seen_nonzero = merge_reservoir(reservoir, seen_nonzero, raw)
            else:
                arr = np.asarray(block)
                sampled_nonzero += int(np.count_nonzero(arr))
                sampled_total += int(arr.size)
                reservoir, seen_nonzero = merge_reservoir(
                    reservoir, seen_nonzero, arr.reshape(-1)
                )

        zero_fraction = (
            1.0 - sampled_nonzero / float(sampled_total)
            if sampled_total
            else None
        )
        if reservoir is None or reservoir.size == 0:
            return np.array([], dtype=float), zero_fraction, "backed_sparse_all_zero_row_sample"
        return reservoir, zero_fraction, "backed_sparse_nonzero_row_sample"

    # Dense ndarray/dataframe/backed dataset. Load all only when bounded.
    if total <= max_values:
        arr = np.asarray(matrix)
        return arr.reshape(-1), float(np.mean(arr == 0)), "dense_full"

    # Use a bounded rectangular sample. This avoids the previous behavior of
    # loading ``max_rows * n_cols`` values before reducing to ``max_values``.
    target = min(max_values, total)
    row_count = min(n_rows, max_rows, target)
    col_count = min(n_cols, max(1, target // row_count))
    row_idx = np.sort(_sample_indices(n_rows, row_count, rng))
    col_idx = np.sort(_sample_indices(n_cols, col_count, rng))

    if isinstance(matrix, pd.DataFrame):
        block = matrix.iloc[row_idx, col_idx].to_numpy()
    else:
        try:
            # Works efficiently for NumPy, memmap, zarr, and many lazy arrays.
            block = np.asarray(matrix[np.ix_(row_idx, col_idx)])
        except Exception:
            # h5py-style datasets generally allow fancy indexing on only one
            # axis. Iterate over whichever sampled axis is smaller.
            pieces: List[np.ndarray] = []
            if col_idx.size <= row_idx.size:
                for j in col_idx:
                    pieces.append(np.asarray(matrix[row_idx, int(j)]).reshape(-1, 1))
                block = np.concatenate(pieces, axis=1) if pieces else np.empty((row_idx.size, 0))
            else:
                for i in row_idx:
                    pieces.append(np.asarray(matrix[int(i), col_idx]).reshape(1, -1))
                block = np.concatenate(pieces, axis=0) if pieces else np.empty((0, col_idx.size))

    flat = np.asarray(block).reshape(-1)
    if flat.size > max_values:  # Defensive guard for unusual backends.
        flat = take_sample(flat, max_values)
    zero_fraction = float(np.mean(flat == 0)) if flat.size else None
    return flat, zero_fraction, "dense_bounded_entry_sample"

def _numeric_summary(values: np.ndarray) -> Dict[str, Any]:
    q = np.quantile(values, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    tolerance = 1e-6
    integer_like = np.isclose(values, np.rint(values), atol=tolerance, rtol=0.0)
    return {
        "min": float(q[0]),
        "q01": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q99": float(q[5]),
        "max": float(q[6]),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "negative_fraction": float(np.mean(values < 0)),
        "positive_fraction": float(np.mean(values > 0)),
        "integer_like_fraction": float(np.mean(integer_like)),
    }


def _infer_matrix_state(values: np.ndarray) -> Dict[str, Any]:
    """Conservative matrix-state heuristic; never treated as a definitive label."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"state": "unknown", "confidence": "low", "reason": "no finite sampled values"}

    nonzero_values = values[values != 0]
    if nonzero_values.size == 0:
        return {
            "state": "empty_or_all_zero",
            "confidence": "medium",
            "reason": "all sampled finite values are zero",
        }

    # Exact zeros are uninformative for distinguishing raw counts from
    # log-normalized values and otherwise inflate integer-like fractions.
    neg = float(np.mean(nonzero_values < 0))
    int_like = float(
        np.mean(np.isclose(nonzero_values, np.rint(nonzero_values), atol=1e-6, rtol=0.0))
    )
    q99 = float(np.quantile(nonzero_values, 0.99))
    vmax = float(np.max(nonzero_values))
    mean = float(np.mean(values))
    std = float(np.std(values))

    if neg == 0.0 and int_like >= 0.985:
        confidence = "high" if vmax >= 20 or q99 >= 5 else "medium"
        return {
            "state": "count_like",
            "confidence": confidence,
            "reason": "nonnegative and almost all sampled values are integer-like",
        }
    if neg >= 0.01 and std > 0 and abs(mean) <= max(0.25, 0.25 * std):
        return {
            "state": "centered_or_scaled",
            "confidence": "medium",
            "reason": "contains negative values and sampled mean is near zero relative to spread",
        }
    if neg == 0.0 and int_like < 0.95 and q99 <= 15:
        confidence = "medium" if q99 <= 10 else "low"
        return {
            "state": "log_or_continuous_nonnegative",
            "confidence": confidence,
            "reason": "nonnegative, predominantly non-integer, and compressed dynamic range",
        }
    if neg > 0:
        return {
            "state": "continuous_with_negative_values",
            "confidence": "medium",
            "reason": "sample contains negative values but is not clearly centered/scaled",
        }
    return {
        "state": "continuous_nonnegative_or_mixed",
        "confidence": "low",
        "reason": "sample does not match a high-confidence count/log/scaled pattern",
    }


def _report_matrix_profile(name: str, p: Dict[str, Any], c: _Collector, category: str) -> None:
    sid = _safe_id(name)
    if not p.get("present"):
        c.add(
            f"MAT_{sid}_MISSING", "ERROR", category, f"Matrix '{name}' is missing",
            "The matrix value is None.",
            "Restore the matrix or explicitly document an intentionally matrix-free AnnData object.",
        )
        return

    if not p.get("shape_matches", False):
        c.add(
            f"MAT_{sid}_SHAPE", "ERROR", category, f"Matrix '{name}' has an invalid shape",
            f"Observed {p.get('shape')}; expected {p.get('expected_shape')}.",
            "Rebuild the AnnData object so the matrix matches obs and var axes.",
        )
    else:
        c.add(
            f"MAT_{sid}_SHAPE", "PASS", category, f"Matrix '{name}' shape is consistent",
            f"Shape is {tuple(p.get('shape', []))}.",
        )

    if "sampling_error" in p:
        c.add(
            f"MAT_{sid}_READ", "WARN", category, f"Could not sample matrix '{name}'",
            p["sampling_error"],
            "Check that the backing file is accessible and the matrix backend supports row slicing.",
        )
        return

    if p.get("numeric") is False:
        c.add(
            f"MAT_{sid}_NUMERIC", "ERROR", category, f"Matrix '{name}' is not numeric",
            f"Observed dtype: {p.get('dtype')}.",
            "Convert expression values to a numeric matrix.",
        )
        return

    finite_fraction = p.get("finite_fraction")
    if finite_fraction is not None and finite_fraction < 1.0:
        severity = "ERROR" if finite_fraction < 0.999 else "WARN"
        c.add(
            f"MAT_{sid}_FINITE", severity, category, f"Matrix '{name}' contains non-finite values",
            f"Estimated finite fraction: {finite_fraction:.4%}.",
            "Trace and remove NaN/Inf values before normalization, dimensionality reduction, or clustering.",
        )

    state = p.get("state_inference", {})
    c.add(
        f"MAT_{sid}_STATE", "INFO", category, f"Heuristic state for matrix '{name}'",
        f"Inferred '{state.get('state', 'unknown')}' with {state.get('confidence', 'low')} confidence. "
        f"This is a heuristic, not proof of preprocessing history.",
        "Compare this inference with the recorded workflow and source files.",
        {
            "reason": state.get("reason", ""),
            "dtype": p.get("dtype"),
            "min": p.get("min"),
            "q99": p.get("q99"),
            "max": p.get("max"),
            "negative_fraction": p.get("negative_fraction"),
            "integer_like_fraction": p.get("integer_like_fraction"),
            "zero_fraction_estimate": p.get("zero_fraction_estimate"),
            "sampling_mode": p.get("sampling_mode"),
            "sample_size": p.get("sample_size"),
        },
    )


def _check_zero_axes(
    matrix: Any,
    n_obs: int,
    n_vars: int,
    c: _Collector,
    rng: np.random.Generator,
    max_rows: int,
) -> None:
    if matrix is None or n_obs == 0 or n_vars == 0:
        return
    try:
        if sparse.issparse(matrix):
            csr = matrix.tocsr()
            zero_obs = int(np.sum(np.diff(csr.indptr) == 0))
            csc = matrix.tocsc()
            zero_vars = int(np.sum(np.diff(csc.indptr) == 0))
            exact = True
        else:
            exact = n_obs <= max_rows and n_obs * n_vars <= 20_000_000
            if exact:
                arr = np.asarray(matrix)
                zero_obs = int(np.sum(np.count_nonzero(arr, axis=1) == 0))
                zero_vars = int(np.sum(np.count_nonzero(arr, axis=0) == 0))
            else:
                rows = np.sort(_sample_indices(n_obs, min(max_rows, n_obs), rng))
                arr = np.asarray(matrix[rows, :])
                zero_obs = int(np.sum(np.count_nonzero(arr, axis=1) == 0))
                # A variable absent in sampled rows is not necessarily globally zero; do not flag as exact.
                zero_vars = -1

        if zero_obs > 0:
            if exact:
                detail = f"Found {zero_obs:,} all-zero observations ({zero_obs / n_obs:.2%})."
            else:
                detail = f"Found {zero_obs:,} all-zero observations in a sample of {min(max_rows, n_obs):,} rows."
            c.add(
                "MAT_ZERO_OBS", "WARN", "matrix", "All-zero observations detected",
                detail,
                "Remove truly empty observations or restore their source counts before analysis.",
            )
        else:
            c.add(
                "MAT_ZERO_OBS", "PASS", "matrix", "No all-zero observations detected",
                "No all-zero observations were found in the checked rows.",
            )

        if zero_vars > 0:
            c.add(
                "MAT_ZERO_VAR", "WARN", "matrix", "All-zero variables detected",
                f"Found {zero_vars:,} all-zero variables ({zero_vars / n_vars:.2%}).",
                "Remove uninformative variables unless they are deliberately retained for panel completeness.",
            )
        elif zero_vars == 0:
            c.add(
                "MAT_ZERO_VAR", "PASS", "matrix", "No all-zero variables detected",
                "No all-zero variables were found.",
            )
    except Exception as exc:
        c.add(
            "MAT_ZERO_AXIS_READ", "INFO", "matrix", "Skipped all-zero axis check",
            f"The matrix backend could not be evaluated safely: {type(exc).__name__}: {exc}",
        )


# -----------------------------------------------------------------------------
# Layers, raw, and annotations
# -----------------------------------------------------------------------------


def _check_layers(
    adata: Any,
    c: _Collector,
    rng: np.random.Generator,
    *,
    max_matrix_sample: int,
    max_rows: int,
) -> None:
    for key, matrix in list(adata.layers.items()):
        name = f"layers[{key!r}]"
        p = _profile_matrix(
            matrix,
            rng=rng,
            max_values=max_matrix_sample,
            max_rows=max_rows,
            expected_shape=(adata.n_obs, adata.n_vars),
            name=name,
        )
        _report_matrix_profile(name, p, c, category="layers")

        key_l = str(key).lower()
        state = p.get("state_inference", {}).get("state")
        neg = p.get("negative_fraction", 0.0) or 0.0
        int_like = p.get("integer_like_fraction", 1.0)
        if any(token in key_l for token in ("count", "raw")):
            if neg > 0:
                c.add(
                    f"LAYER_{_safe_id(key)}_COUNT_NEG", "ERROR", "layers",
                    f"Count-designated layer '{key}' contains negative values",
                    f"Estimated negative fraction: {neg:.2%}.",
                    "Restore nonnegative raw counts or rename the layer to reflect its actual content.",
                )
            if int_like is not None and int_like < 0.98:
                c.add(
                    f"LAYER_{_safe_id(key)}_COUNT_NONINT", "WARN", "layers",
                    f"Count-designated layer '{key}' is not integer-like",
                    f"Estimated integer-like fraction: {int_like:.2%}; inferred state: {state}.",
                    "Verify whether this layer contains raw counts, expected counts, or normalized values.",
                )

    if len(adata.layers) == 0:
        c.add(
            "LAYER_NONE", "INFO", "layers", "No layers present",
            "The object stores no alternate expression matrices in .layers.",
            "For reproducibility, consider retaining raw counts in layers['counts'] when appropriate.",
        )


def _check_raw(
    adata: Any,
    c: _Collector,
    rng: np.random.Generator,
    *,
    max_matrix_sample: int,
    max_rows: int,
) -> None:
    raw = getattr(adata, "raw", None)
    if raw is None:
        c.add(
            "RAW_NONE", "INFO", "raw", "No .raw snapshot present",
            "The object does not contain an AnnData.raw snapshot.",
            "This is valid if raw counts or preprocessed values are stored and documented elsewhere.",
        )
        return

    if int(raw.n_obs) != int(adata.n_obs):
        c.add(
            "RAW_OBS", "ERROR", "raw", ".raw observation count is inconsistent",
            f"raw.n_obs={raw.n_obs:,}, adata.n_obs={adata.n_obs:,}.",
            "Recreate .raw from the correctly aligned observations.",
        )
    else:
        c.add(
            "RAW_OBS", "PASS", "raw", ".raw observation count is consistent",
            f"Both contain {adata.n_obs:,} observations.",
        )

    raw_names = pd.Index(raw.var_names).astype(str)
    current_names = pd.Index(adata.var_names).astype(str)
    raw_dup = int(raw_names.duplicated().sum())
    if raw_dup:
        c.add(
            "RAW_VAR_DUP", "ERROR", "raw", "Duplicate variable names in .raw",
            f"Found {raw_dup:,} duplicated raw variable identifiers.",
            "Resolve duplicate identifiers before using .raw for marker extraction.",
        )

    overlap = len(current_names.intersection(raw_names))
    fraction = overlap / max(1, len(current_names))
    if fraction < 0.95:
        c.add(
            "RAW_VAR_OVERLAP", "WARN", "raw", "Low variable overlap between .raw and current .var",
            f"Only {overlap:,}/{len(current_names):,} current variables ({fraction:.1%}) occur in .raw.",
            "Confirm that .raw originates from the same feature namespace and preprocessing lineage.",
        )
    else:
        c.add(
            "RAW_VAR_OVERLAP", "PASS", "raw", "Current variables are represented in .raw",
            f"Overlap is {overlap:,}/{len(current_names):,} ({fraction:.1%}).",
        )

    p = _profile_matrix(
        raw.X,
        rng=rng,
        max_values=max_matrix_sample,
        max_rows=max_rows,
        expected_shape=(adata.n_obs, raw.n_vars),
        name="raw.X",
    )
    _report_matrix_profile("raw.X", p, c, category="raw")


def _check_var_annotations(adata: Any, c: _Collector) -> None:
    var = adata.var
    names = pd.Index(adata.var_names).astype(str)

    if "highly_variable" in var.columns:
        hv = var["highly_variable"]
        if not pd.api.types.is_bool_dtype(hv.dtype):
            unique = set(hv.dropna().unique().tolist())
            if unique.issubset({0, 1, True, False}):
                c.add(
                    "VAR_HVG_DTYPE", "INFO", "var", "highly_variable is not boolean dtype",
                    f"Observed dtype {hv.dtype}, values appear binary.",
                    "Convert to bool for unambiguous filtering.",
                )
            else:
                c.add(
                    "VAR_HVG_DTYPE", "WARN", "var", "Invalid highly_variable values",
                    f"Observed dtype {hv.dtype} with non-binary values.",
                    "Recompute or correct the highly_variable annotation.",
                    {"examples": list(unique)[:10]},
                )
        hv_count = int(pd.Series(hv).fillna(False).astype(bool).sum())
        if hv_count == 0:
            c.add(
                "VAR_HVG_EMPTY", "WARN", "var", "No highly variable genes selected",
                "The highly_variable column exists but contains no True values.",
                "Recompute HVGs or remove the stale column.",
            )
        elif hv_count == adata.n_vars:
            c.add(
                "VAR_HVG_ALL", "INFO", "var", "All variables are marked highly variable",
                "This may be intentional for targeted panels but is unusual for whole-transcriptome data.",
            )
        else:
            c.add(
                "VAR_HVG_COUNT", "PASS", "var", "Highly variable gene annotation is populated",
                f"{hv_count:,}/{adata.n_vars:,} variables are marked highly variable.",
            )

    # Ensembl versions and collisions after version stripping.
    ens_version_mask = names.str.match(r"^ENS[A-Z]*G\d+\.\d+$")
    ens_plain_mask = names.str.match(r"^ENS[A-Z]*G\d+$")
    if int(ens_version_mask.sum()) > 0:
        stripped = names.str.replace(r"\.\d+$", "", regex=True)
        collisions = int(stripped.duplicated().sum())
        c.add(
            "VAR_ENSEMBL_VERSION", "INFO", "var", "Versioned Ensembl identifiers detected",
            f"{int(ens_version_mask.sum()):,} variables include a dot-version suffix.",
            "Record the reference annotation release; strip versions only with a traceable mapping.",
            {"collisions_after_stripping": collisions},
        )
        if collisions:
            c.add(
                "VAR_ENSEMBL_COLLISION", "WARN", "var", "Ensembl version stripping would create collisions",
                f"Removing dot-version suffixes would create {collisions:,} duplicated identifiers.",
                "Do not strip suffixes blindly; resolve one-to-many or duplicated identifiers explicitly.",
            )

    # Gene symbol case conventions: heuristic only.
    sample = names[: min(len(names), 50_000)]
    human_like = int(sample.str.match(r"^[A-Z0-9][A-Z0-9._-]*$").sum())
    mouse_like = int(sample.str.match(r"^[A-Z][a-z0-9][A-Za-z0-9._-]*$").sum())
    denom = max(1, len(sample))
    if human_like / denom > 0.25 and mouse_like / denom > 0.25 and not (ens_version_mask.any() or ens_plain_mask.any()):
        c.add(
            "VAR_SPECIES_CASE_MIX", "INFO", "var", "Mixed gene-symbol capitalization patterns",
            "Both human-like uppercase and mouse-like title-case symbols are common. This can also arise "
            "from non-gene features, so the result is heuristic.",
            "Confirm species and feature namespace before merging datasets or calculating mitochondrial QC.",
            {"human_like_fraction": human_like / denom, "mouse_like_fraction": mouse_like / denom},
        )

    mt_upper = int(names.str.startswith("MT-").sum())
    mt_lower = int(names.str.startswith("mt-").sum())
    if mt_upper > 0 and mt_lower > 0:
        c.add(
            "VAR_MT_CASE_MIX", "WARN", "var", "Mixed mitochondrial gene prefixes",
            f"Found {mt_upper} 'MT-' and {mt_lower} 'mt-' identifiers.",
            "Confirm whether species or naming conventions were mixed before mitochondrial QC.",
        )

    # Duplicates after case-folding are dangerous in cross-species or case-insensitive joins.
    folded = names.str.casefold()
    folded_dup = int(folded.duplicated().sum())
    if folded_dup and int(names.duplicated().sum()) == 0:
        c.add(
            "VAR_CASE_COLLISION", "WARN", "var", "Variable names collide after case-folding",
            f"Found {folded_dup:,} case-insensitive identifier collisions.",
            "Avoid case-insensitive joins until species and identifier conventions are resolved.",
        )


def _check_obs_annotations(adata: Any, c: _Collector) -> None:
    obs = adata.obs

    # Common QC columns: range and semantic checks.
    for col in ("total_counts", "n_counts", "n_genes_by_counts", "n_genes"):
        if col in obs.columns and pd.api.types.is_numeric_dtype(obs[col]):
            vals = pd.to_numeric(obs[col], errors="coerce")
            if bool((vals.dropna() < 0).any()):
                c.add(
                    f"OBS_{_safe_id(col)}_NEG", "ERROR", "obs",
                    f"QC column '{col}' contains negative values",
                    "Count-derived QC metrics should not be negative.",
                    "Recompute the QC column from the intended count matrix.",
                )

    pct_cols = [col for col in obs.columns if str(col).lower().startswith("pct_") or "percent" in str(col).lower()]
    for col in pct_cols:
        if pd.api.types.is_numeric_dtype(obs[col]):
            vals = pd.to_numeric(obs[col], errors="coerce").dropna()
            if len(vals) and ((vals < 0).any() or (vals > 100).any()):
                c.add(
                    f"OBS_{_safe_id(col)}_RANGE", "WARN", "obs",
                    f"Percentage-like column '{col}' lies outside 0–100",
                    f"Observed range: {float(vals.min()):.4g} to {float(vals.max()):.4g}.",
                    "Confirm whether values are fractions (0–1), percentages (0–100), or corrupted.",
                )

    # Candidate cluster labels with a single level are often stale or uninformative.
    cluster_tokens = ("cluster", "leiden", "louvain", "domain", "cell_type", "celltype")
    for col in obs.columns:
        low = str(col).lower()
        if any(token in low for token in cluster_tokens):
            n_unique = int(obs[col].nunique(dropna=True))
            if n_unique <= 1 and len(obs) > 1:
                c.add(
                    f"OBS_CLUSTER_SINGLE_{_safe_id(col)}", "WARN", "obs",
                    f"Label column '{col}' has {n_unique} observed class",
                    "A clustering/annotation-like column is constant or empty.",
                    "Remove the stale label or verify that the analysis intentionally produced one group.",
                )
            elif n_unique > max(1000, int(0.5 * len(obs))):
                c.add(
                    f"OBS_CLUSTER_HIGHCARD_{_safe_id(col)}", "INFO", "obs",
                    f"Label column '{col}' has very high cardinality",
                    f"Observed {n_unique:,} labels for {len(obs):,} observations.",
                    "Confirm that this is a label column rather than a unique identifier.",
                )


def _check_multidimensional_annotations(adata: Any, c: _Collector, rng: np.random.Generator) -> None:
    for slot, mapping, expected_first in (
        ("obsm", adata.obsm, adata.n_obs),
        ("varm", adata.varm, adata.n_vars),
    ):
        for key, value in list(mapping.items()):
            shape = tuple(getattr(value, "shape", ()))
            if not shape or int(shape[0]) != int(expected_first):
                c.add(
                    f"{slot.upper()}_{_safe_id(key)}_SHAPE", "ERROR", slot,
                    f"{slot}['{key}'] has inconsistent first dimension",
                    f"Observed shape {shape}; expected first dimension {expected_first:,}.",
                    f"Recompute or remove {slot}['{key}'].",
                )
                continue

            arr_sample = _sample_array(value, rng=rng, max_values=100_000)
            if arr_sample is not None and np.issubdtype(arr_sample.dtype, np.number):
                finite = np.isfinite(arr_sample.astype(float, copy=False))
                if finite.size and not bool(finite.all()):
                    c.add(
                        f"{slot.upper()}_{_safe_id(key)}_FINITE", "WARN", slot,
                        f"{slot}['{key}'] contains non-finite values",
                        f"Estimated non-finite fraction: {1 - float(finite.mean()):.4%}.",
                        f"Recompute or sanitize {slot}['{key}'] before downstream use.",
                    )

            key_l = str(key).lower()
            if slot == "obsm" and ("pca" in key_l or "latent" in key_l or "embed" in key_l):
                if len(shape) != 2 or shape[1] < 2:
                    c.add(
                        f"OBSM_{_safe_id(key)}_DIM", "WARN", "obsm",
                        f"Embedding '{key}' has fewer than two dimensions",
                        f"Observed shape {shape}.",
                        "Confirm that this key contains an embedding rather than a score vector.",
                    )
                else:
                    try:
                        block = np.asarray(value[: min(expected_first, 5000), :], dtype=float)
                        var = np.nanvar(block, axis=0)
                        zero_dims = int(np.sum(var <= 1e-12))
                        if zero_dims:
                            c.add(
                                f"OBSM_{_safe_id(key)}_ZEROVAR", "WARN", "obsm",
                                f"Embedding '{key}' contains zero-variance dimensions",
                                f"Found {zero_dims}/{shape[1]} near-constant dimensions in the checked rows.",
                                "Recompute the embedding or remove degenerate dimensions.",
                            )
                    except Exception:
                        pass


def _check_pairwise_matrices(adata: Any, c: _Collector) -> None:
    for slot, mapping, expected in (
        ("obsp", adata.obsp, adata.n_obs),
        ("varp", adata.varp, adata.n_vars),
    ):
        for key, matrix in list(mapping.items()):
            shape = tuple(getattr(matrix, "shape", ()))
            if shape != (expected, expected):
                c.add(
                    f"{slot.upper()}_{_safe_id(key)}_SHAPE", "ERROR", slot,
                    f"{slot}['{key}'] is not square/aligned",
                    f"Observed shape {shape}; expected ({expected}, {expected}).",
                    f"Recompute or remove {slot}['{key}'].",
                )
                continue

            key_l = str(key).lower()
            try:
                if sparse.issparse(matrix):
                    data = np.asarray(matrix.data, dtype=float)
                    if data.size and not np.isfinite(data).all():
                        c.add(
                            f"{slot.upper()}_{_safe_id(key)}_FINITE", "ERROR", slot,
                            f"{slot}['{key}'] contains non-finite values",
                            "Sparse data values contain NaN or Inf.",
                            f"Recompute {slot}['{key}'].",
                        )
                    if "connect" in key_l and data.size and np.min(data) < 0:
                        c.add(
                            f"{slot.upper()}_{_safe_id(key)}_NEG", "WARN", slot,
                            f"Connectivity matrix '{key}' contains negative weights",
                            f"Minimum stored value is {float(np.min(data)):.4g}.",
                            "Confirm that signed edges are intentional; many graph algorithms expect nonnegative weights.",
                        )
                    if "connect" in key_l or "distance" in key_l or "adj" in key_l:
                        diff = matrix - matrix.T
                        asym = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
                        if asym > 1e-6:
                            c.add(
                                f"{slot.upper()}_{_safe_id(key)}_ASYM", "WARN", slot,
                                f"Pairwise matrix '{key}' is asymmetric",
                                f"Maximum stored |A-A.T| is {asym:.4g}.",
                                "Confirm whether a directed graph is intended; otherwise symmetrize or recompute it.",
                            )
                    if slot == "obsp" and "connect" in key_l:
                        degree = np.asarray(matrix.getnnz(axis=1)).ravel()
                        isolated = int(np.sum(degree == 0))
                        if isolated:
                            c.add(
                                f"OBSP_{_safe_id(key)}_ISOLATED", "WARN", "obsp",
                                f"Graph '{key}' contains isolated observations",
                                f"Found {isolated:,}/{expected:,} nodes with no stored edges.",
                                "Check filtering, coordinate validity, and neighborhood parameters.",
                            )
                else:
                    arr = np.asarray(matrix)
                    if np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
                        c.add(
                            f"{slot.upper()}_{_safe_id(key)}_FINITE", "ERROR", slot,
                            f"{slot}['{key}'] contains non-finite values",
                            "Dense pairwise matrix contains NaN or Inf.",
                            f"Recompute {slot}['{key}'].",
                        )
                    if expected <= 10_000 and ("connect" in key_l or "distance" in key_l or "adj" in key_l):
                        asym = float(np.nanmax(np.abs(arr - arr.T)))
                        if asym > 1e-6:
                            c.add(
                                f"{slot.upper()}_{_safe_id(key)}_ASYM", "WARN", slot,
                                f"Pairwise matrix '{key}' is asymmetric",
                                f"Maximum |A-A.T| is {asym:.4g}.",
                                "Confirm whether a directed graph is intended.",
                            )
            except Exception as exc:
                c.add(
                    f"{slot.upper()}_{_safe_id(key)}_READ", "INFO", slot,
                    f"Could not fully inspect {slot}['{key}']",
                    f"{type(exc).__name__}: {exc}",
                )


# -----------------------------------------------------------------------------
# Analysis and spatial checks
# -----------------------------------------------------------------------------


def _check_neighbors_metadata(adata: Any, c: _Collector) -> None:
    neighbors = adata.uns.get("neighbors") if isinstance(adata.uns, Mapping) else None
    if neighbors is None:
        if any("connect" in str(k).lower() for k in adata.obsp.keys()):
            c.add(
                "NEIGH_META_MISSING", "WARN", "neighbors", "Neighbor graph metadata is missing",
                "Connectivity-like matrices exist in .obsp but .uns['neighbors'] is absent.",
                "Store graph parameters and key references in .uns['neighbors'] for provenance.",
            )
        return
    if not isinstance(neighbors, Mapping):
        c.add(
            "NEIGH_META_TYPE", "ERROR", "neighbors", ".uns['neighbors'] is not mapping-like",
            f"Observed type: {type(neighbors).__name__}.",
            "Recompute neighbor metadata using a standard AnnData/Scanpy-compatible structure.",
        )
        return

    con_key = neighbors.get("connectivities_key", "connectivities")
    dist_key = neighbors.get("distances_key", "distances")
    for role, key in (("connectivities", con_key), ("distances", dist_key)):
        if key not in adata.obsp:
            c.add(
                f"NEIGH_{role.upper()}_MISSING", "ERROR", "neighbors",
                f"Neighbor metadata references missing {role}",
                f".uns['neighbors'] points to .obsp['{key}'], but that key does not exist.",
                "Recompute neighbors or repair the stale key reference.",
            )

    params = neighbors.get("params")
    if not isinstance(params, Mapping) or not params:
        c.add(
            "NEIGH_PARAMS_MISSING", "WARN", "neighbors", "Neighbor parameters are missing",
            ".uns['neighbors']['params'] is absent or empty.",
            "Record n_neighbors, metric, representation, method, and random state when available.",
        )
    else:
        if "n_neighbors" not in params:
            c.add(
                "NEIGH_N_MISSING", "INFO", "neighbors", "n_neighbors is not recorded",
                "The graph metadata does not record neighborhood size.",
            )
        if not any(k in params for k in ("random_state", "seed")):
            c.add(
                "NEIGH_SEED_MISSING", "INFO", "neighbors", "Neighbor random seed is not recorded",
                "A deterministic seed is not present in neighbor metadata.",
                "Record the seed when the graph construction method is stochastic.",
            )


def _check_analysis_metadata(adata: Any, x_profile: Dict[str, Any], c: _Collector) -> None:
    state = x_profile.get("state_inference", {}).get("state")
    has_log1p = isinstance(adata.uns, Mapping) and "log1p" in adata.uns
    if has_log1p and state == "count_like":
        c.add(
            "META_LOG1P_STALE", "WARN", "provenance", "log1p metadata conflicts with count-like X",
            ".uns['log1p'] exists, while X appears count-like from sampled values.",
            "Confirm whether X was replaced after normalization; remove stale metadata or restore the intended matrix.",
        )
    elif not has_log1p and state == "log_or_continuous_nonnegative":
        c.add(
            "META_LOG1P_ABSENT", "INFO", "provenance", "Transformation metadata is absent",
            "X appears log-transformed or otherwise continuous/nonnegative, but .uns['log1p'] is absent.",
            "Record transformation details, input layer, base, and pseudocount in provenance metadata.",
        )

    # PCA metadata and embedding coherence.
    has_pca = "X_pca" in adata.obsm
    pca_uns = adata.uns.get("pca") if isinstance(adata.uns, Mapping) else None
    pcs = adata.varm.get("PCs") if "PCs" in adata.varm else None
    if has_pca and pca_uns is None:
        c.add(
            "META_PCA_MISSING", "INFO", "provenance", "PCA embedding lacks .uns['pca'] metadata",
            "X_pca exists but PCA parameters/variance metadata are absent.",
            "Record PCA settings and explained variance for reproducibility.",
        )
    if has_pca and pcs is not None:
        try:
            n_scores = int(adata.obsm["X_pca"].shape[1])
            n_loadings = int(pcs.shape[1])
            if n_scores != n_loadings:
                c.add(
                    "META_PCA_DIM", "ERROR", "provenance", "PCA scores and loadings have different dimensions",
                    f"X_pca has {n_scores} components; varm['PCs'] has {n_loadings}.",
                    "Recompute PCA or remove stale scores/loadings.",
                )
        except Exception:
            pass


def _check_spatial(adata: Any, c: _Collector) -> None:
    spatial_keys = [k for k in adata.obsm.keys() if str(k).lower() == "spatial"]
    if not spatial_keys:
        if isinstance(adata.uns, Mapping) and "spatial" in adata.uns:
            c.add(
                "SPAT_COORD_MISSING", "ERROR", "spatial", "Spatial metadata exists but coordinates are missing",
                ".uns['spatial'] exists, but .obsm['spatial'] is absent.",
                "Restore observation-aligned coordinates in adata.obsm['spatial'].",
            )
        else:
            c.add(
                "SPAT_NONE", "INFO", "spatial", "No spatial coordinates detected",
                "The object does not contain .obsm['spatial']; spatial-specific checks were skipped.",
            )
        return

    coords_obj = adata.obsm[spatial_keys[0]]
    try:
        coords = np.asarray(coords_obj)
    except Exception as exc:
        c.add(
            "SPAT_COORD_READ", "ERROR", "spatial", "Could not read spatial coordinates",
            f"{type(exc).__name__}: {exc}",
            "Store coordinates as a numeric n_obs × 2 or n_obs × 3 array.",
        )
        return

    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        c.add(
            "SPAT_COORD_SHAPE", "ERROR", "spatial", "Invalid spatial coordinate shape",
            f"Observed shape {coords.shape}; expected n_obs × 2 (or more dimensions).",
            "Restore an observation-aligned coordinate matrix.",
        )
        return
    if coords.shape[1] > 3:
        c.add(
            "SPAT_COORD_DIM", "INFO", "spatial", "Spatial coordinate array has more than three columns",
            f"Observed {coords.shape[1]} coordinate columns.",
            "Document the meaning and units of each column.",
        )

    try:
        xy = coords[:, :2].astype(float)
    except Exception:
        c.add(
            "SPAT_COORD_NUMERIC", "ERROR", "spatial", "Spatial coordinates are not numeric",
            f"Observed dtype: {coords.dtype}.",
            "Convert coordinates to numeric pixel or physical units.",
        )
        return

    finite_rows = np.isfinite(xy).all(axis=1)
    bad = int((~finite_rows).sum())
    if bad:
        c.add(
            "SPAT_COORD_FINITE", "ERROR", "spatial", "Spatial coordinates contain NaN or Inf",
            f"Found {bad:,}/{len(xy):,} invalid coordinate rows.",
            "Repair or exclude invalid observations before constructing spatial graphs.",
        )
    valid = xy[finite_rows]
    if len(valid) == 0:
        return

    unique_count = int(np.unique(valid, axis=0).shape[0])
    duplicates = len(valid) - unique_count
    if duplicates:
        frac = duplicates / len(valid)
        severity = "WARN" if frac < 0.05 else "ERROR"
        c.add(
            "SPAT_COORD_DUP", severity, "spatial", "Duplicate spatial coordinates detected",
            f"{duplicates:,}/{len(valid):,} valid rows duplicate an earlier x/y coordinate ({frac:.2%}).",
            "Confirm whether duplicated locations represent legitimate multi-molecule/cell records or a merge error.",
        )

    ranges = np.ptp(valid, axis=0)
    if np.any(ranges == 0):
        c.add(
            "SPAT_COORD_DEGENERATE", "ERROR", "spatial", "Degenerate spatial coordinate axis",
            f"Coordinate ranges are x={ranges[0]:.4g}, y={ranges[1]:.4g}.",
            "Check whether x/y were loaded correctly.",
        )
    else:
        aspect = float(max(ranges) / min(ranges))
        if aspect > 100:
            c.add(
                "SPAT_COORD_ASPECT", "WARN", "spatial", "Extreme spatial coordinate aspect ratio",
                f"Coordinate range aspect ratio is {aspect:.1f}:1.",
                "Check coordinate units, x/y columns, scale factors, and accidental flattening.",
            )

        centered = valid - valid.mean(axis=0, keepdims=True)
        if len(valid) >= 3:
            svals = np.linalg.svd(centered, compute_uv=False)
            if len(svals) >= 2 and svals[0] > 0 and svals[1] / svals[0] < 1e-6:
                c.add(
                    "SPAT_COORD_COLLINEAR", "WARN", "spatial", "Spatial coordinates are nearly collinear",
                    "The second singular value is negligible relative to the first.",
                    "Confirm that a two-dimensional tissue layout was not collapsed to a line.",
                )

    # Visium/Scanpy-style image metadata.
    spatial_uns = adata.uns.get("spatial") if isinstance(adata.uns, Mapping) else None
    if spatial_uns is None:
        c.add(
            "SPAT_UNS_MISSING", "INFO", "spatial", "Coordinates lack .uns['spatial'] image metadata",
            "Spatial coordinates are present, but no Scanpy/Visium-style image metadata was found.",
            "This is valid for platforms without tissue images; otherwise retain images, scale factors, and units.",
        )
        return
    if not isinstance(spatial_uns, Mapping):
        c.add(
            "SPAT_UNS_TYPE", "ERROR", "spatial", ".uns['spatial'] is not mapping-like",
            f"Observed type: {type(spatial_uns).__name__}.",
            "Reconstruct the spatial metadata dictionary.",
        )
        return

    libraries = list(spatial_uns.keys())
    if len(libraries) > 1:
        lib_cols = [col for col in adata.obs.columns if str(col).lower() in {"library_id", "library", "sample", "sample_id"}]
        if not lib_cols:
            c.add(
                "SPAT_MULTI_LIB_AMBIG", "WARN", "spatial", "Multiple spatial libraries lack an observation mapping",
                f".uns['spatial'] contains {len(libraries)} libraries, but no obvious library/sample column exists in .obs.",
                "Add an observation-level library identifier before validating or plotting multi-library coordinates.",
            )

    for lib_id, lib in spatial_uns.items():
        if not isinstance(lib, Mapping):
            c.add(
                f"SPAT_LIB_{_safe_id(lib_id)}_TYPE", "WARN", "spatial",
                f"Spatial library '{lib_id}' metadata is not mapping-like",
                f"Observed type: {type(lib).__name__}.",
            )
            continue
        images = lib.get("images", {})
        scalefactors = lib.get("scalefactors", {})
        if not isinstance(images, Mapping) or len(images) == 0:
            continue
        if not isinstance(scalefactors, Mapping):
            scalefactors = {}

        for image_key, image in images.items():
            shape = tuple(getattr(image, "shape", ()))
            if len(shape) < 2:
                c.add(
                    f"SPAT_IMG_{_safe_id(lib_id)}_{_safe_id(image_key)}_SHAPE", "WARN", "spatial",
                    f"Image '{lib_id}/{image_key}' has invalid shape",
                    f"Observed shape {shape}.",
                    "Store an H×W grayscale image or H×W×C color image.",
                )
                continue
            height, width = int(shape[0]), int(shape[1])
            scale_key = f"tissue_{image_key}_scalef"
            scale = scalefactors.get(scale_key, 1.0)
            try:
                scale = float(scale)
            except Exception:
                c.add(
                    f"SPAT_SCALE_{_safe_id(lib_id)}_{_safe_id(image_key)}_TYPE", "WARN", "spatial",
                    f"Scale factor '{scale_key}' is not numeric",
                    f"Observed value: {scale!r}.",
                    "Store scale factors as finite positive numbers.",
                )
                continue
            if not np.isfinite(scale) or scale <= 0:
                c.add(
                    f"SPAT_SCALE_{_safe_id(lib_id)}_{_safe_id(image_key)}_VALUE", "ERROR", "spatial",
                    f"Scale factor '{scale_key}' is invalid",
                    f"Observed value: {scale}.",
                    "Restore a finite positive full-resolution-to-image scale factor.",
                )
                continue

            scaled = valid * scale
            inside = (
                (scaled[:, 0] >= 0)
                & (scaled[:, 0] < width)
                & (scaled[:, 1] >= 0)
                & (scaled[:, 1] < height)
            )
            coverage = float(inside.mean())
            if coverage < 0.5:
                c.add(
                    f"SPAT_IMG_{_safe_id(lib_id)}_{_safe_id(image_key)}_BOUNDS", "WARN", "spatial",
                    f"Most coordinates fall outside image '{lib_id}/{image_key}'",
                    f"Only {coverage:.1%} of valid x/y coordinates lie within image bounds after scale={scale:g}.",
                    "Check x/y order, image orientation, coordinate units, library assignment, and scale factors.",
                    {"image_shape": list(shape), "scale_key": scale_key, "scale": scale},
                )
            else:
                c.add(
                    f"SPAT_IMG_{_safe_id(lib_id)}_{_safe_id(image_key)}_BOUNDS", "PASS", "spatial",
                    f"Coordinates are compatible with image '{lib_id}/{image_key}' bounds",
                    f"{coverage:.1%} of valid coordinates lie inside the scaled image bounds.",
                )


def _check_color_metadata(adata: Any, c: _Collector) -> None:
    if not isinstance(adata.uns, Mapping):
        return
    for key, value in list(adata.uns.items()):
        key_s = str(key)
        if not key_s.endswith("_colors"):
            continue
        obs_key = key_s[: -len("_colors")]
        if obs_key not in adata.obs.columns:
            c.add(
                f"COLOR_{_safe_id(obs_key)}_ORPHAN", "INFO", "provenance",
                f"Color metadata '{key_s}' has no matching obs column",
                f"adata.obs does not contain '{obs_key}'.",
                f"Remove stale adata.uns['{key_s}'] or restore the matching label column.",
            )
            continue
        try:
            n_colors = len(value)
        except Exception:
            continue
        s = adata.obs[obs_key]
        n_categories = len(s.cat.categories) if isinstance(s.dtype, pd.CategoricalDtype) else int(s.nunique(dropna=True))
        if n_colors != n_categories:
            c.add(
                f"COLOR_{_safe_id(obs_key)}_LEN", "WARN", "provenance",
                f"Color metadata length does not match labels for '{obs_key}'",
                f"Found {n_colors} colors but {n_categories} categories/labels.",
                "Regenerate the color palette after filtering, relabeling, or removing unused categories.",
            )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _is_sparse_dataset(matrix: Any) -> bool:
    name = type(matrix).__name__.lower()
    module = type(matrix).__module__.lower()
    return (
        "sparsedataset" in name
        or "csrdataset" in name
        or "cscdataset" in name
        or ("anndata" in module and hasattr(matrix, "to_memory") and hasattr(matrix, "format"))
    )


def _sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=int)
    return rng.choice(n, size=k, replace=False).astype(int)


def _sample_array(value: Any, rng: np.random.Generator, max_values: int) -> Optional[np.ndarray]:
    try:
        shape = tuple(int(x) for x in value.shape)
    except Exception:
        return None
    if not shape or math.prod(shape) == 0:
        return np.array([])
    try:
        if math.prod(shape) <= max_values:
            return np.asarray(value)
        if len(shape) == 1:
            idx = np.sort(_sample_indices(shape[0], min(shape[0], max_values), rng))
            return np.asarray(value[idx])
        rows = min(shape[0], max(1, max_values // max(1, math.prod(shape[1:]))))
        idx = np.sort(_sample_indices(shape[0], rows, rng))
        return np.asarray(value[idx, ...])
    except Exception:
        return None


def _safe_id(value: Any) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value).upper()).strip("_")
    if not text:
        text = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8].upper()
    return text[:80]


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None

