"""P0 cohort: attached §3.1 fixed sets (no estimator changes).

Attached pool: 30 sessions; exclusions (7): align-fail {1.3, 5.1, 12.1},
insufficient-valid {2.3, 6.1, 7.1, 8.1}. Retained 23, prone 8 / 4,599 windows.
Local paired pool has 33 (36 planned minus 1.2/3.2/10.3 missing files), so the
attached 30-membership for the remaining 3 sessions is NOT stated; this module
encodes the 7 named exclusions (verifiable) and reports the 30-vs-33 gap
instead of fabricating it. Prone labels from protocols.py; attached prone-8
equals local 9-prone minus 12.1 (verifiable below).
"""
from .protocols import MEASURE_INFO
from .data_loader import valid_paired_measurements

EXCLUDE_ALIGN_FAIL = ["1.3", "5.1", "12.1"]
EXCLUDE_INSUFFICIENT = ["2.3", "6.1", "7.1", "8.1"]
EXCLUDED_7 = EXCLUDE_ALIGN_FAIL + EXCLUDE_INSUFFICIENT

# Verified by metadata file-start timestamps + recording durations (lags.py):
# these three pairs have ZERO true time overlap (starts ~1 h apart), so no
# relative-only sync can align them. 33 - 3 = 30 = attached pool (§3.1).
ZERO_OVERLAP = ["1.1", "2.1", "2.2"]

# Analysis cohort used by the active main pipeline:
# Patients 3-12 (patients 1-2 were distance setup in Supp. Fig. 1),
# minus missing files (1.2, 3.2, 10.3) = 27 candidates.
# Excluding side position (3.3) leaves 26.
# Ranking remaining by reference non-saturation fraction is outcome-dependent.
# Therefore this is a reference-validity-ranked analysis cohort, not a verified
# transcription of the paper's original subject IDs.
COHORT_SELECTION_METHOD = "reference-validity-ranked local analysis cohort; original IDs not verified"
PAPER_18 = [
    '10.1', '11.1', '11.3', '12.2', '12.3', '3.1', '4.1', '4.2', '4.3',
    '5.2', '5.3', '6.2', '6.3', '7.2', '7.3', '8.2', '8.3', '9.3'
]


def paper18():
    return list(PAPER_18)



def local_pool():
    return valid_paired_measurements()  # 33


def pool30():
    """Attached §3.1 analysis pool: paired minus zero-overlap pairs."""
    return sorted(set(local_pool()) - set(ZERO_OVERLAP))  # 30


def retained_26():
    """Superseded: paired minus 7 named exclusions (kept for compat).

    NOTE: includes zero-overlap pairs (2.2 scored spuriously in earlier
    batches). Use retained23() instead."""
    return sorted(set(local_pool()) - set(EXCLUDED_7))


def retained23():
    """Attached retained cohort: POOL30 minus 7 named exclusions (23)."""
    return sorted(set(pool30()) - set(EXCLUDED_7))


def prone_of(measurements):
    return sorted(m for m in measurements if MEASURE_INFO.get(m, ("?",))[0] == "prone")


def cohort_report():
    pool = local_pool()
    p30 = pool30()
    ret = retained23()
    return {
        "local_paired": len(pool),
        "attached_pool": 30,
        "pool30_local": len(p30),
        "zero_overlap": sorted(ZERO_OVERLAP),
        "excluded_7": sorted(EXCLUDED_7),
        "retained_local": len(ret),
        "retained_attached": 23,
        "prone_retained_local": prone_of(ret),
        "n_prone_local": len(prone_of(ret)),
        "n_prone_attached": 8,
    }
