"""Handles experiment dataset."""

from pathlib import Path
from tkinter.filedialog import askdirectory, askopenfilename

import numpy as np
import wvfreader as wvf
import xarray as xr
from matplotlib import pyplot as plt
from rich.console import Console

__all__ = ["create_dataset"]

_TO_MICROSECOND: int = 1_000_000  # conversion factor from [s] to [µs]


def create_dataset(
    wvf_path: Path | None = None,
    video_dir: Path | None = None,
    mask_dir: Path | None = None,
    wireframe_path: Path | None = None,
    save_path: Path | None = None,
    sample_rate: int = 1_000_000,
    video_fps: int = 100_000,
):
    """Create a dataset from the waveform, video, and mask files.

    This function creates a dataset from the waveform, video, and mask files.
    The `trigger`, `II_gate`, `Vf_A`, `Vf_D`, `V_probe`, and `I_sat` signals
    are extracted from CH1, CH3, CH4, CH8, CH9, and CH10 traces in waveform file, respectively.

    .. note::
        The minimum time step of dataset is assumed to be 1 µs (i.e., 1 MHz sample rate),
        and `time`-related coordinates are represented as `int64` in microseconds to avoid
        floating point precision issues.

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
    sample_rate : int, optional
        The sample rate of the waveform data in Hz, by default 1 MHz.
    video_fps : int, optional
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

        # Convert time variable to int64 in microseconds
        t0: int = round(time[0] * _TO_MICROSECOND)  # start time in µs
        time = np.arange(t0, t0 + len(time), _TO_MICROSECOND / sample_rate, dtype=np.int64)

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
                        units="µs",
                        long_name="Time",
                        sample_rate=sample_rate,
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
        if len(video_files) == 0:
            raise FileNotFoundError(f"No .tif files found in {video_dir}")

        # video time steps
        # NOTE: we assume the video start time is when II_gate first exceeds 2.0 V
        # After that, photons are captured during the I.I.'s exposure time.
        # Ignoring I.I.'s gate delay time (~100 ns).
        t0: int = (
            ds["II_gate"]
            .where(ds["II_gate"] > 2.0, drop=True)
            .where(ds["time"] > 0, drop=True)
            .time[0]
            .data.item()
        )

        # Set the video time steps
        _dt: int = round(_TO_MICROSECOND / video_fps)  # time step in µs
        time = np.arange(t0, t0 + len(video_files) * _dt, _dt, dtype=np.int64)

        # Trim the dataset to match the video time steps
        # NOTE: video start time is later than the wvf start time, so we need to trim the dataset
        # to match the video time steps.
        # ds = ds.where(ds["time"] >= time[0], drop=True)

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
                        "units": "µs",
                        "long_name": "Time",
                        "description": "Video time steps",
                        "sample_rate": video_fps,
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
            console.log(f"Dataset saved to {save_path}")

        return ds_merged
