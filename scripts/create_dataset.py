#!/usr/bin/env -S pixi run python
"""Create dataset from video and waveform data."""

from pathlib import Path

from cherab.nagdis.experiment.dataset import create_dataset

# %%
# Define paths
# -------------------------------------------------------
# Video data directory where the TIFF files are stored.
VIDEO_DATA = Path("/media/koyo/Extreme SSD/20240705/")

# Observer data directory where
OBSERVER_DATA = Path(__file__).parents[1] / "observers" / "data"
MASK_DIR = OBSERVER_DATA / "mask20240705" / "w1280-h192"
WIREFRAME_PATH = OBSERVER_DATA / "wireframe_20240705_w1280_h192.npy"

# %%
# Create dataset
# -------------------------------------------------------
# Parameters for dataset creation
SCCM = 430
WAVE = 589
GAIN = 44

ds = create_dataset(
    wvf_path=Path(f"/media/koyo/Extreme SSD/20240705/waveform/{SCCM:>03}sccm/{WAVE}-1.wvf"),
    video_dir=VIDEO_DATA / f"{SCCM:>03}sccm" / "videos" / f"G{GAIN}_{WAVE}",
    mask_dir=MASK_DIR,
    wireframe_path=WIREFRAME_PATH,
    save_path=Path(f"sccm{SCCM:>03}_G{GAIN}_{WAVE}.nc"),
)
print(ds)
