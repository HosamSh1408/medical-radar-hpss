"""
Config for Beltrao et al. 2022 reproduction.
Paper: Contactless radar-based breathing monitoring of premature infants in NICU.
All values from paper unless marked [INFERRED].
"""
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parents[2]
BELTRAO_DATA = BASE / "beltrao_data"
RADAR_DIR = BELTRAO_DATA / "radar"
REF_DIR = BELTRAO_DATA / "reference"

# Radar hardware (Methods: Radar system)
FC_HZ = 24e9
LAMBDA_M = 3e8 / FC_HZ  # ~12.5 mm
LAMBDA_MM = LAMBDA_M * 1000
FS_RADAR = 16  # Hz, ADC sampling

# Reference device (Methods: Reference cabled device)
FS_REF_RAW = 256  # Resp waveform Hz
FS_REF_RATE = 1   # RespRate Hz
REF_RES_BPM = 1

# Processing windows (Experimental results)
WIN_S = 30
OVERLAP_S = 28
UPDATE_S = WIN_S - OVERLAP_S  # 2 s
WIN_SAMPLES = WIN_S * FS_RADAR  # 480

# STFT for NMF (Methods: NMF, rectangular 3 s, 2 s overlap, zeropad 256)
STFT_WIN_S = 3
STFT_OVERLAP_S = 2
STFT_HOP_S = STFT_WIN_S - STFT_OVERLAP_S  # 1 s
STFT_WIN = STFT_WIN_S * FS_RADAR  # 48
STFT_HOP = STFT_HOP_S * FS_RADAR  # 16
STFT_NFFT = 256

# NMF
NMF_K = 11  # paper empirical
NMF_MAX_ITER = 500  # [INFERRED]
NMF_RANDOM_STATE = 0
NMF_ALPHA_H = 0.0  # L1/L2 sparsity pressure on H; 0 = unregularized baseline
NMF_L1_RATIO = 0.0  # 0 = L2 only, 1.0 = pure L1 (elementwise sparsity)
NMF_PRESET = "beltrao_tuned"  # or "attached_locked" (raw V, tol 1e-4, 2000it, Hoyer>0.55 & max>2.mean)
RBM_ALPHA = 3.0  # strict threshold: prevents over-pruning genuine breathing ridges on clean windows
RBM_SPARSITY_FRAC = 0.6  # top-3 time frames must hold >60% of row energy to flag as impulsive RBM
MIN_MOVE_MAXBASES = 2  # <=2 bases excluded = minimal movement (paper Fig.7)

# Bandpass (Breathing rate estimation)
BAND_LOW_HZ = 0.3
BAND_HIGH_HZ = 3.0
KAISER_BETA = 6.5
FIR_TAPS = 64  # [INFERRED]

# NLS (Methods: NLS, Lk=2, +-5 bpm)
NLS_LK = 2
NLS_RADIUS_BPM = 5.0
NLS_GRID_STEP_BPM = 0.1  # [INFERRED]
NLS_NFFT = 4096  # [INFERRED] for PSD interpolation

# Evaluation
ACC_THRESHOLDS = [3, 6, 10]
SYNC_MAX_LAG_S = 120  # [INFERRED]
