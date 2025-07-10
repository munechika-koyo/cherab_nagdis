"""Handles experiment dataset."""

from pathlib import Path
from tkinter.filedialog import askdirectory, askopenfilename

import numpy as np
import wvfreader as wvf
import xarray as xr
from matplotlib import pyplot as plt
from rich.console import Console

__all__ = ["create_dataset"]


def create_dataset(
    wvf_path: Path | None = None,
    video_dir: Path | None = None,
    mask_dir: Path | None = None,
    wireframe_path: Path | None = None,
    save_path: Path | None = None,
    video_fps: float = 100e3,
):
    """Create a dataset from the waveform, video, and mask files.

    This function creates a dataset from the waveform, video, and mask files.
    The `trigger`, `II_gate`, `Vf_A`, `Vf_D`, `V_probe`, and `I_sat` signals
    are extracted from CH1, CH3, CH4, CH8, CH9, and CH10 traces in waveform file, respectively.

    Parameters
    ----------
    wvf_path : Path, optional
        The path to the `*.wvf` waveform file. If None, a file dialog will be opened.
    video_dir : Path, optional
        The path to the directory containing the `*.tif` files. If None, a directory dialog will be
        opened.
    mask_dir : Path, optional
        The path to the directory containing the `*.npy` mask files. If None, a directory dialog
        will be opened.
    wireframe_path : Path, optional
        The path to the wireframe image file. If None, the wireframe image will not be added.
    save_path : Path, optional
        The path to save the dataset as a NetCDF file. If None, the dataset will not be saved.
    video_fps : float, optional
        The video frame rate in Hz, by default 100 kHz.
    """
    console = Console()
    with console.status("[bold green]Creating dataset..."):
        # === Load the wvf file ===
        if wvf_path is None:
            wvf_path = Path(askopenfilename(title="Select the .wvf file"))
        wvf_data = wvf.datafile(str(wvf_path))
        wvf_data = wvf_data[wvf_path.stem]

        time = wvf_data.traces["CH1"].t.ravel()
        trigger = wvf_data.traces["CH1"].y.ravel()
        II_gate = wvf_data.traces["CH3"].y.ravel()
        Vf_A = wvf_data.traces["CH4"].y.ravel()
        Vf_D = wvf_data.traces["CH8"].y.ravel()
        V_probe = wvf_data.traces["CH9"].y.ravel()
        I_sat = wvf_data.traces["CH10"].y.ravel() * 10

        ds = xr.Dataset(
            {
                "trigger": ("time", trigger, dict(units="V", long_name="Trigger signal")),
                "II_gate": (
                    "time",
                    II_gate,
                    dict(units="V", long_name="Image Intensifier gate signal"),
                ),
                "Vf_A": ("time", Vf_A, dict(units="V", long_name="mid scanning A Vf lower")),
                "Vf_D": ("time", Vf_D, dict(units="V", long_name="mid scanning D Vf upper")),
                "V_probe": ("time", V_probe, dict(units="V", long_name="Probe voltage")),
                "I_sat": ("time", I_sat, dict(units="mA", long_name="Ion saturation current")),
            },
            coords={
                "time": (
                    "time",
                    time,
                    dict(
                        units="s",
                        long_name="Time",
                        sample_freaquency=1 / (time[1] - time[0]),
                    ),
                )
            },
            attrs={
                "wvf_path": str(wvf_path),
                "date": wvf_data.date,
            },
        )
        console.log(f"Loaded waveform data from {wvf_path}")

        # === Load the mask files ===
        if mask_dir is None:
            mask_dir = Path(
                askdirectory(title="Select the mask directory including mask .npy files")
            )
        else:
            if not mask_dir.exists():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        mask_files = sorted(mask_dir.glob("*.npy"))
        masks = {}
        for mask_file in mask_files:
            mask = np.load(mask_file)
            masks[mask_file.stem] = mask

        console.log(f"Loaded {len(masks)} mask files from {mask_dir}")

        # === Load the video files ===
        if video_dir is None:
            video_dir = Path(askdirectory(title="Select the video directory"))

        video_files = sorted(video_dir.glob("*.tif"))

        # video time steps
        t0 = (
            ds["II_gate"]
            .where(ds["II_gate"] > 2.0, drop=True)
            .where(ds["time"] > 0, drop=True)
            .time[0]
        )
        dt = 1 / video_fps
        t0 += 0.5 * dt  # start from the middle of the first frame
        time = np.linspace(t0, t0 + len(video_files) / video_fps, len(video_files))

        # Trim the dataset to match the video time steps
        # NOTE: video start time is later than the wvf start time, so we need to trim the dataset
        # to match the video time steps.
        ds = ds.where(ds["time"] >= time[0], drop=True)

        # Align end time by trimming either the wvf dataset or the video files
        if time[-1] > ds["time"][-1]:
            time = time[time <= ds["time"][-1]]
            video_files = video_files[: len(time)]
        else:
            ds = ds.where(ds["time"] <= time[-1], drop=True)

        videos = {
            port: np.zeros((len(time), np.count_nonzero(mask))) for port, mask in masks.items()
        }
        for i, video_file in enumerate(video_files):
            image = np.rot90(plt.imread(str(video_file)), k=2)
            for port, mask in masks.items():
                videos[port][i] = image[mask]

        ds_video = xr.Dataset(
            {
                port: (
                    ["time_video", f"pixel-{port}"],
                    video,
                    dict(units="DN", long_name="Digital number"),
                )
                for port, video in videos.items()
            }
            | {
                f"mask-{port}": (["y", "x"], mask, dict(long_name=f"{port} mask array"))
                for port, mask in masks.items()
            },
            coords={
                "time_video": (
                    "time_video",
                    time,
                    {
                        "units": "s",
                        "long_name": "Time",
                        "description": "Video time steps",
                        "sample_frequency": 1 / (time[1] - time[0]),
                    },
                )
            }
            | {f"pixel-{port}": np.arange(np.count_nonzero(mask)) for port, mask in masks.items()}
            | {
                "height": (
                    "y",
                    np.arange(masks["port-1"].shape[0]),
                    {"units": "pixel", "long_name": "Height pixel index"},
                ),
                "width": (
                    "x",
                    np.arange(masks["port-1"].shape[1]),
                    {"units": "pixel", "long_name": "Width pixel index"},
                ),
            },
            attrs={
                "video_dir": str(video_dir),
            },
        )
        console.log(f"Loaded video data from {video_dir}")

        # Add CAD wireframe image
        if isinstance(wireframe_path, Path) and wireframe_path.exists():
            ds_video = ds_video.assign(
                {
                    "wireframe": (
                        ["y", "x", "rgba"],
                        np.load(wireframe_path),
                        dict(long_name="CAD wireframe image"),
                    )
                }
            )
            console.log(f"Loaded wireframe image from {wireframe_path}")

        # === Merge the datasets ===
        ds_merged = xr.merge([ds, ds_video], combine_attrs="drop_conflicts")

        # === Save the dataset ===
        if save_path is not None:
            ds_merged.to_netcdf(save_path, format="NETCDF4")
            console.log(f"[Dataset saved to {save_path}")

        return ds_merged
