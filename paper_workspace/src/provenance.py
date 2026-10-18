"""Cache provenance and freshness checks for the active analysis pipeline.

The NPZ files are analysis products, not source data.  A cache is reusable only
when it was produced by the same active code, configuration, input files, and
runtime dependency versions.  This prevents stale results from silently
appearing in tables and figures after an implementation change.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from . import config as C
from .data_loader import MEASURE_TO_RADAR, MEASURE_TO_REF

PIPELINE_VERSION = "main-v3-audit-fix-1"

_SRC_DIR = Path(__file__).resolve().parent
_CORE_FILES = (
    "config.py", "cohort.py", "data_loader.py", "demod.py", "evaluation.py",
    "estimation.py", "filtering.py", "hpss.py", "lags.py", "rbm_nmf.py",
    "run_all.py", "run_batch.py", "run_measurement.py",
    "provenance.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "pyedflib")
    return {name: metadata.version(name) for name in names if _installed(name)}


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def _input_manifest(measurement: str) -> dict[str, str]:
    if measurement not in MEASURE_TO_RADAR or measurement not in MEASURE_TO_REF:
        raise ValueError(f"Measurement {measurement!r} has no paired radar/reference input")
    radar = C.RADAR_DIR / f"{MEASURE_TO_RADAR[measurement]}.csv"
    reference = C.REF_DIR / f"{MEASURE_TO_REF[measurement]}.edf"
    missing = [str(p) for p in (radar, reference) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))
    return {"radar_csv": _sha256(radar), "reference_edf": _sha256(reference)}


def expected_provenance(measurement: str) -> dict:
    """Return the complete provenance record expected for a cache."""
    source_hashes = {
        name: _sha256(_SRC_DIR / name)
        for name in _CORE_FILES
    }
    return {
        "pipeline_version": PIPELINE_VERSION,
        "measurement": measurement,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "source_sha256": source_hashes,
        "input_sha256": _input_manifest(measurement),
    }


def provenance_json(measurement: str) -> str:
    return json.dumps(expected_provenance(measurement), sort_keys=True, separators=(",", ":"))


def cache_is_current(path: Path, measurement: str) -> bool:
    """Validate required arrays and exact provenance without trusting mtime."""
    try:
        with __import__("numpy").load(path, allow_pickle=False) as z:
            required = {"est", "est_dft", "est_nls", "est_hpss", "ref", "valid",
                        "minmove", "nmf_converged", "lag", "corr", "t_nmf", "t_hpss",
                        "provenance_json"}
            if not required.issubset(set(z.files)):
                return False
            stored = str(z["provenance_json"].item())
        return stored == provenance_json(measurement)
    except Exception:
        return False
