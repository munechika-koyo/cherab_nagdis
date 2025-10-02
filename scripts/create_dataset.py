#!/usr/bin/env -S pixi run python
"""Create dataset from video and waveform data."""

from pathlib import Path

from cherab.nagdis.experiment import create_dataset

# %%
# Define paths
# -------------------------------------------------------
ROOT = Path(__file__).parents[1] / "output"
EXPERIMENT = ROOT / "experimental" / "20240703"
VIDEO_DIR = EXPERIMENT / "videos"
WAVEFORM_DIR = EXPERIMENT / "waveform"
MASK_DIR = ROOT / "camera_calibration" / "mask20240705" / "w1280-h192"
WIREFRAME_PATH = ROOT / "camera_calibration" / "wireframe_20240705_w1280_h192.npy"

# %%
# Create dataset
# -------------------------------------------------------
# Parameters for dataset creation
SCCM = 430
gain_waves = [
    (32, 589),
    (40, 450),
    (53, 405),
    (87, 372),
]

for gain, wave in gain_waves:
    print(f"Creating dataset (SCCM: {SCCM:>03}, Gain: {gain:>02}, Wave: {wave:>03})...")
    ds = create_dataset(
        wvf_path=WAVEFORM_DIR / f"{SCCM:>03}sccm" / f"{wave:>03}-1.wvf",
        video_dir=VIDEO_DIR / f"G{gain}_{wave}",
        mask_dir=MASK_DIR,
        wireframe_path=WIREFRAME_PATH,
        save_path=ROOT / f"sccm{SCCM:>03}_G{gain}_{wave}.nc",
    )
