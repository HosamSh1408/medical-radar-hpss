"""Phase demodulation: CSD (Eq.9, adopted) + AD (Eq.8, Fig.3 comparison)."""
import numpy as np
from . import config as C


def csd(bI, bQ):
    """Complex signal demodulation. x = bI+j*bQ with mean (DC) removed per window."""
    bI = np.asarray(bI, dtype=float)
    bQ = np.asarray(bQ, dtype=float)
    x = bI + 1j * bQ
    x = x - np.mean(x)
    return x


def ad(bI, bQ, lam=C.LAMBDA_M):
    """Arctangent demodulation (Eq.8). UNUSED by the active pipeline (audit A10).

    The active pipeline adopts CSD (csd, above); this function is retained only
    for the Fig.3 clean-segment illustration. Fragile; not used for scoring.
    """
    bI = np.asarray(bI, dtype=float)
    bQ = np.asarray(bQ, dtype=float)
    bI0 = bI - np.mean(bI)
    bQ0 = bQ - np.mean(bQ)
    phase = np.unwrap(np.arctan2(bQ0, bI0))
    return lam / (4 * np.pi) * phase
