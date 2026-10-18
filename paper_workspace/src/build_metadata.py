"""Build auditable local cohort metadata without selecting by performance."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from . import config as C
from .cohort import PAPER_18, EXCLUDED_7, ZERO_OVERLAP, pool30, retained23
from .data_loader import MEASURE_TO_RADAR, MEASURE_TO_REF, valid_paired_measurements
from .protocols import MEASURE_INFO

OUT = Path(__file__).resolve().parents[1] / "metadata" / "cohort_ledger.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_label(measurement: str) -> str:
    return "annotations.pdf transcription in protocols.py"



def build_rows():
    paired = set(valid_paired_measurements())
    p30 = set(pool30())
    r23 = set(retained23())
    excluded = set(EXCLUDED_7)
    zero_overlap = set(ZERO_OVERLAP)
    rows = []
    for m in sorted(paired, key=lambda x: tuple(int(p) for p in x.split("."))):
        position, twins, lid = MEASURE_INFO.get(m, ("unknown", False, "unknown"))
        radar = C.RADAR_DIR / f"{MEASURE_TO_RADAR[m]}.csv"
        reference = C.REF_DIR / f"{MEASURE_TO_REF[m]}.edf"
        rows.append({
            "measurement": m,
            "patient_group": m.split(".")[0],
            "position": position,
            "cobedding_twins": str(bool(twins)).lower(),
            "lid_status": lid,
            "radar_file": str(radar.relative_to(C.BASE)),
            "reference_file": str(reference.relative_to(C.BASE)),
            "radar_sha256": sha256(radar),
            "reference_sha256": sha256(reference),
            "paired_local": "true",
            "in_current_paper18": str(m in PAPER_18).lower(),
            "in_pool30_no_zero_overlap": str(m in p30).lower(),
            "in_retained23": str(m in r23).lower(),
            "named_exclusion": str(m in excluded).lower(),
            "zero_overlap": str(m in zero_overlap).lower(),
            "protocol_evidence": source_label(m),
            "cohort_evidence": (
                "reference-validity-ranked local selection; original ID not verified"
                if m in PAPER_18 else
                "local paired intersection and explicit exclusion logic"
            ),
        })
    return rows


def main():
    rows = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} paired measurements)")


if __name__ == "__main__":
    main()
