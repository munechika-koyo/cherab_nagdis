#!/usr/bin/env -S pixi run python
"""Restore Radiant dataset from video and waveform data."""

from pathlib import Path

import xarray as xr

DATA_DIR = Path(__file__).parents[1] / "data"

# %%
# Load Transmittance
# -------------------------------------------------------
ds_transmittance = xr.open_dataset(DATA_DIR / "transmittance_avg.nc")
print(ds_transmittance)

# %%
# Load Gain coefficients
# -------------------------------------------------------
ds_gain = xr.open_dataset(DATA_DIR / "gain_factor.nc")
print(ds_gain)

# %%
# Load Data
# -------------------------------------------------------
tasks = [
    ("n=3", 589, 32),
    ("n=4", 450, 40),
    ("n=5", 405, 53),
    ("n=7", 372, 87),
]
dt = 1e-6  # Exposure time in seconds

for transient, λ, gain in tasks:
    path = DATA_DIR / "sccm430" / f"{transient}_G{gain}" / "ca_s23_ma10.nc"
    ds = xr.open_dataset(path)

    # ds_new = ds.copy()

    # Transform the pixel data to radiant flux
    for port in range(1, 5 + 1):
        # Determine the window transmittance based on the port
        match port:
            case 1:
                window = "Window1"
            case 2:
                window = "Window2"
            case _:
                window = "CleanWindow"

        # Get the raw data for the port
        data = ds[f"port-{port}"].data

        # Apply the gain factor and transmittance correction
        data = (
            (data / dt)  # D/Δt
            * ds_gain["factor"].sel(λ=λ, G=gain).data  # Φ/f
            * (1 / ds_transmittance["transmittance"].sel(λ=λ, window=window).data)  # 1/T
        )

        # Store the corrected data back in the dataset
        ds[f"port-{port}"].loc[:, :] = data

    # Save the modified dataset
    ds.to_netcdf(path=path.with_stem("ca_s23_ma10_flux"))

    print(f"Processed and saved: {path.with_stem('ca_s23_ma10_flux')}")
