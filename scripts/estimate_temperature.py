#!/usr/bin/env -S pixi run python
"""Estimate electron temperature from tomographic reconstructions.

This script estimates the electron temperature using tomographic reconstructions from two different
spectral lines of Helium I. It then creates an animation showing the spatial distribution of the
estimated temperature and overlays emissivity contours.
"""

from pathlib import Path

import matplotlib
import numpy as np
import ultraplot as uplt
import xarray as xr
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from rich.progress import Progress

matplotlib.use("Agg")

ROOT = Path(__file__).parents[1]

# %%
# Load data
# =========

tomograms = {
    3: ROOT / "data/sccm430/n=3_G32/tomograms/2025_10_06_23_28_03.nc",
    4: ROOT / "data/sccm430/n=4_G40/tomograms/2025_10_06_22_44_41.nc",
    5: ROOT / "data/sccm430/n=5_G53/tomograms/2025_10_06_23_43_24.nc",
    7: ROOT / "data/sccm430/n=7_G87/tomograms/2025_10_07_07_29_10.nc",
}

ds_atomic = xr.open_dataset(ROOT / "data" / "atomic_data_he1.nc")

# %%
# Estimate temperature
# =====================
# Using lines 5 and 7
i, j = 5, 7

temp = (ds_atomic.E.sel(i=i) - ds_atomic.E.sel(i=j)) / np.log(
    (ds_atomic.g.sel(i=i) / ds_atomic.g.sel(i=j))
    * (ds_atomic.A.sel(i=i) / ds_atomic.A.sel(i=j))
    * (ds_atomic.wavelength.sel(i=j) / ds_atomic.wavelength.sel(i=i))
    * (
        xr.open_dataset(tomograms[j]).reconstructions
        / xr.open_dataset(tomograms[i]).reconstructions
    )
)

# Load tomogram for emissivity
ds_tomo = xr.open_dataset(tomograms[i])


# Limit area
r_max = 26e-3  # [m]
temp_limit = temp.where(temp.X**2 + temp.Y**2 < r_max**2, drop=True)
emiss_limit = ds_tomo.reconstructions.where(ds_tomo.X**2 + ds_tomo.Y**2 < r_max**2, drop=True)

# Assign coords
temp_limit = temp_limit.assign_coords(X=temp_limit.X * 1e3, Y=temp_limit.Y * 1e3)
emiss_limit = emiss_limit.assign_coords(X=emiss_limit.X * 1e3, Y=emiss_limit.Y * 1e3)

# Set color limits
vmin_temp = temp_limit.where(temp_limit > 0).min(dim=["X", "Y", "tau"])
vmax_temp = temp_limit.max(dim=["X", "Y", "tau"])

vmin_emiss = emiss_limit.min(dim=["X", "Y", "tau"])
vmax_emiss = emiss_limit.max(dim=["X", "Y", "tau"])

ports = [4, 5]

# %%
# Plot temperature and emissivity
# ===============================

fig, axs = uplt.subplots(
    nrows=1,
    ncols=len(ports),
    tight=False,  # NOTE: set tight=False to avoid changing layout during animation
)

axs.format(
    xlabel="$x$ [mm]",
    ylabel="$y$ [mm]",
    xlocator=10,
    ylocator=10,
    xminorlocator=2,
    yminorlocator=2,
)

mappables_temp = []
contours_emiss = []

# Plot for initial tau
for i_port in range(len(ports)):
    ax = axs[i_port]

    # Select dataarray
    da_temp = temp_limit.sel(port=ports[i_port]).isel(tau=0)
    da_emiss = emiss_limit.sel(port=ports[i_port]).isel(tau=0)

    # Plot — Temperature
    mappables_temp.append(
        ax.pcolormesh(
            da_temp.T,
            cmap="turbo",
            vmax=vmax_temp.sel(port=ports[i_port]).item(),
            vmin=vmin_temp.sel(port=ports[i_port]).item(),
            discrete=False,
        )
    )

    # Plot - Emissivity
    contour = ax.contour(
        da_emiss.T,
        cmap="inferno",
        levels=10,
        vmax=vmax_emiss.sel(port=ports[i_port]).item(),
        vmin=vmin_emiss.sel(port=ports[i_port]).item(),
    )
    contours_emiss.append(contour)

    # Colorbar — Temperature
    ax.colorbar(
        mappables_temp[i_port],
        label="$T_\\mathrm{e}$ [eV]",
        loc="t",
        tickdir="out",
        pad=0.2,
        width=0.15,
    )

    # Colorbar - Emissivity
    ax.colorbar(
        ScalarMappable(
            cmap="inferno",
            norm=Normalize(
                vmin=vmin_emiss.sel(port=ports[i_port]).item(),
                vmax=vmax_emiss.sel(port=ports[i_port]).item(),
            ),
        ),
        label="Emissivity, $ɛ_{52}$ [W/m³]",
        loc="b",
        tickdir="out",
        width=0.15,
    )

    ax.format(
        aspect="equal",
        ultitle=f"#{ports[i_port]}",
        urtitle=f"{da_temp.tau:+.0f} µs",
    )


def update(i_tau: int, maps_temp: list, axs: list) -> list:
    """Update function for animation.

    Parameters
    ----------
    i_tau
        Current frame index for the animation.
    maps_temp
        List of temperature mappable objects to update.
    axs
        List of axes objects for plotting.

    Returns
    -------
    list
        Updated list of temperature mappable objects.
    """
    global contours_emiss

    for i, (port, map_temp, contour) in enumerate(
        zip(ports, maps_temp, contours_emiss, strict=True),
    ):
        # Update Temp
        map_temp.set(array=temp_limit.sel(port=port).isel(tau=i_tau).T.data)

        # Update tau title
        axs[i].format(
            urtitle=f"{temp_limit.tau.isel(tau=i_tau):+.0f} µs",
        )

        # Remove old contours
        contour.remove()

        # Plot new contours
        _contour = axs[i].contour(
            emiss_limit.sel(port=port).isel(tau=i_tau).T,
            cmap="inferno",
            levels=10,
            vmax=vmax_emiss.sel(port=port).item(),
            vmin=vmin_emiss.sel(port=port).item(),
        )
        contours_emiss[i] = _contour
    return maps_temp


# Create animation
num_frames = temp_limit.tau.size
ani = FuncAnimation(
    fig,
    update,
    fargs=(
        mappables_temp,
        axs,
    ),
    frames=num_frames,
    interval=15,
    blit=False,
    repeat_delay=15,
)

# Save animation
with Progress() as progress:
    task = progress.add_task("[cyan]Saving frame...", total=num_frames)
    ani.save(
        str(ROOT / "data/sccm430/temperature_emission.mp4"),
        writer="ffmpeg",
        fps=60,
        progress_callback=lambda i, n: progress.update(task, advance=1),
    )
