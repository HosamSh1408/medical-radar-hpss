"""Master's thesis experimental framework (WP0-WP5).

Strictly decoupled from the frozen student baseline (`paper_workspace/src/`,
commit `6b19fdd`, tag `v1.0-student-baseline`) so reading `.npz` caches and
raw `beltrao_data/` telemetry never mutates SHA-256 baseline provenance.
"""

__all__ = ["io_utils", "edf_extractor", "exp0_forensics"]
__version__ = "0.1.0-wp0"
