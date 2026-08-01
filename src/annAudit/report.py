from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2, "PASS": 3}
__version__ = '1.0.0'

@dataclass(frozen=True)
class Finding:
    """One audit finding."""

    check_id: str
    severity: str
    category: str
    title: str
    detail: str
    recommendation: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sev = self.severity.upper()
        if sev not in SEVERITY_ORDER:
            raise ValueError(f"Unsupported severity: {self.severity}")
        object.__setattr__(self, "severity", sev)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class AuditReport:
    """Collection of findings and object-level metadata."""

    metadata: Dict[str, Any]
    findings: List[Finding]

    @property
    def counts(self) -> Dict[str, int]:
        return {
            sev: sum(f.severity == sev for f in self.findings)
            for sev in ("ERROR", "WARN", "INFO", "PASS")
        }

    @property
    def exit_code(self) -> int:
        """0=no error/warning, 1=warning only, 2=one or more errors."""
        if self.counts["ERROR"]:
            return 2
        if self.counts["WARN"]:
            return 1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": _json_safe(self.metadata),
            "summary": self.counts,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, path: Optional[os.PathLike[str] | str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def render_text(self, include_pass: bool = False) -> str:
        c = self.counts
        lines = [
            "=" * 60,
            f"annAudit: AnnData audit, Version: {__version__}",
            "MIT License, Aug 1 2026, 22:25, UTC+8",
            "=" * 60,
            f"Shape: {self.metadata.get('n_obs', '?'):,} observations × "
            f"{self.metadata.get('n_vars', '?'):,} variables",
            f"Findings: {c['ERROR']} ERROR, {c['WARN']} WARN, "
            f"{c['INFO']} INFO, {c['PASS']} PASS",
        ]
        if self.metadata.get("matrix_state"):
            ms = self.metadata["matrix_state"]
            lines.append(
                f"X heuristic state: {ms.get('state', 'unknown')} "
                f"(confidence={ms.get('confidence', 'unknown')})"
            )
        lines.append("-")

        selected = [f for f in self.findings if include_pass or f.severity != "PASS"]
        selected.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.category, f.check_id))
        if not selected:
            lines.append("No errors or warnings were found.")
        for idx, f in enumerate(selected, 1):
            lines.append(f"[{f.severity}] {f.check_id} | {f.category} | {f.title}")
            lines.append(f"  {f.detail}")
            if f.recommendation:
                lines.append(f"  Recommendation: {f.recommendation}")
            if f.evidence:
                compact = json.dumps(_json_safe(f.evidence), ensure_ascii=False, sort_keys=True)
                lines.append(f"  Evidence: {compact}")
            if idx != len(selected):
                lines.append("")
        return "\n".join(lines)

    def print(self, include_pass: bool = False, file: Any = None) -> None:
        print(self.render_text(include_pass=include_pass), file=file or sys.stdout)


class _Collector:
    def __init__(self) -> None:
        self.findings: List[Finding] = []

    def add(
        self,
        check_id: str,
        severity: str,
        category: str,
        title: str,
        detail: str,
        recommendation: str = "",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.findings.append(
            Finding(
                check_id=check_id,
                severity=severity,
                category=category,
                title=title,
                detail=detail,
                recommendation=recommendation,
                evidence=evidence or {},
            )
        )

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
