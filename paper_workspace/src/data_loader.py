"""
Data loaders porting MATLAB mapping_logic_radar_to_edf/.
- readRadarRaw.m: radar CSV cols 11,12 (1-indexed) = 'I value','Q value'.
- readReferenceData.m: EDF Resp (256 Hz) + RespRate (1 Hz).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pyedflib
from . import config as C

MEASURE_TO_RADAR = {
    '1.1': '20191120T122632.6722028_1', '1.2': '20191121T123707.8896805_1',
    '1.3': '20191122T122923.3777715_1',
    '2.1': '20191120T131717.1086334_1', '2.2': '20191121T131532.5819625_1',
    '2.3': '20191122T130358.6712441_1',
    '3.1': '20191127T115656.1374257_1', '3.2': '20191128T124343.4409922_1',
    '3.3': '20191129T122744.8856478_1',
    '4.1': '20191129T131320.0818421_1', '4.2': '20191202T121559.9456583_1',
    '4.3': '20191203T121222.2758503_1',
    '5.1': '20191203T130234.5132236_1', '5.2': '20191204T121304.7997448_1',
    '5.3': '20191205T121522.4414096_1',
    '6.1': '20191203T134458.1434142_1', '6.2': '20191204T125408.4225468_1',
    '6.3': '20191205T125503.7694909_1',
    '7.1': '20191210T131043.0243695_1', '7.2': '20191211T121818.8779724_1',
    '7.3': '20191212T121224.9057437_1',
    '8.1': '20191210T122823.3108886_1', '8.2': '20191211T125350.6499072_1',
    '8.3': '20191212T124837.0841098_1',
    '9.1': '20191212T133215.9436564_1', '9.2': '20191213T124658.6486329_1',
    '9.3': '20191216T131729.2892346_1',
    '10.1': '20200109T122737.1167572_1', '10.2': '20200110T122027.9850530_1',
    # 10.3: no radar file
    '11.1': '20200113T125426.4708101_1', '11.2': '20200114T125036.6626953_1',
    '11.3': '20200115T124959.8181414_1',
    '12.1': '20200113T133613.4971846_1', '12.2': '20200114T120919.2159340_1',
    '12.3': '20200115T121155.2635871_1',
}

MEASURE_TO_REF = {
    '1.1': '1.0.0.62.4.1_L20112019_131955',
    # 1.2: no reference
    '1.3': '1.0.0.64.4.3_L22112019_122837',
    '2.1': '2.1.0.62.4.1+2_L20112019_140909', '2.2': '2.1.0.63.4.3_L21112019_141254',
    '2.3': '2.1.0.64.4.4_L22112019_130316',
    '3.1': '3.0.1.9.0.1_L27112019_114941',
    # 3.2: no reference
    '3.3': '3.0.1.11.0.4_L29112019_122702',
    '4.1': '4.0.1.7.0.1_L29112019_131242', '4.2': '4.0.1.10.0.2_L02122019_121513',
    '4.3': '4.0.1.11.0.3_L03122019_121139',
    '5.1': '5.0.0.90.0.1_L03122019_130152', '5.2': '5.0.0.91.0.2_L04122019_121101',
    '5.3': '5.0.0.92.0.3_L05122019_121432',
    '6.1': '6.0.0.90.0.1_L03122019_134416', '6.2': '6.0.0.91.0.2_L04122019_125326',
    '6.3': '6.0.0.92.0.3_L05122019_125426',
    '7.1': '7.0.1.13.0.1_L10122019_131005', '7.2': '7.0.1.14.0.2_L11122019_121730',
    '7.3': '7.0.1.15.0.3_L12122019_121140',
    '8.1': '8.1.1.13.0.1_L10122019_122723', '8.2': '8.1.1.14.0.2_L11122019_125307',
    '8.3': '8.1.1.15.0.3_L12122019_124759',
    '9.1': '9.0.0.16.0.1_L12122019_133139', '9.2': '9.0.0.17.0.2_L13122019_124613',
    '9.3': '9.0.0.20.0.03_L16122019_131640',
    '10.1': '10.1.0.48.0.1_L09012020_122632', '10.2': '10.1.0.49.0.2_L10012020_121724',
    '10.3': '10.1.0.52.0.3_L13012020_120845',
    '11.1': '11.0.0.51.0.1_L13012020_125328', '11.2': '11.0.0.52.0.2_L14012020_124939',
    '11.3': '11.0.0.53.0.3_L15012020_124843',
    '12.1': '12.0.1.27.0.1_L13012020_133515', '12.2': '12.0.1.28.0.2_L14012020_120813',
    '12.3': '12.0.1.29.0.3_L15012020_121051',
}


def load_radar(measurement, radar_dir=None):
    """Return (bI, bQ) as in readRadarRaw.m. fs=16 Hz."""
    radar_dir = Path(radar_dir) if radar_dir else C.RADAR_DIR
    if measurement not in MEASURE_TO_RADAR:
        raise ValueError(f"No radar file for measurement {measurement} (e.g. 10.3)")
    fname = MEASURE_TO_RADAR[measurement] + ".csv"
    f = radar_dir / fname
    df = pd.read_csv(f, skiprows=2)
    bI = df["I value"].to_numpy(dtype=float)
    bQ = df["Q value"].to_numpy(dtype=float)
    return bI, bQ


def load_reference(measurement, ref_dir=None):
    """Return (resp_raw_256Hz, resp_rate_1Hz) as in readReferenceData.m."""
    ref_dir = Path(ref_dir) if ref_dir else C.REF_DIR
    if measurement not in MEASURE_TO_REF:
        raise ValueError(f"No reference file for measurement {measurement}")
    fname = MEASURE_TO_REF[measurement] + ".edf"
    f = str(ref_dir / fname)
    r = pyedflib.EdfReader(f)
    try:
        labels = [r.getLabel(i) for i in range(r.signals_in_file)]
        # Find Resp and RespRate indices robustly
        try:
            i_resp = labels.index("Resp")
        except ValueError:
            i_resp = 1
        try:
            i_rate = labels.index("RespRate")
        except ValueError:
            i_rate = 8
        resp_raw = r.readSignal(i_resp).astype(float)
        resp_rate = r.readSignal(i_rate).astype(float)
        fs_raw = r.getSampleFrequency(i_resp)
        fs_rate = r.getSampleFrequency(i_rate)
    finally:
        r._close()
    return resp_raw, resp_rate, float(fs_raw), float(fs_rate)


def valid_paired_measurements():
    return sorted(set(MEASURE_TO_RADAR) & set(MEASURE_TO_REF))
