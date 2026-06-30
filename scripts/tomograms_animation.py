#!/usr/bin/env -S pixi run python
"""Create animation of tomographic reconstructions.

This script loads tomographic reconstruction data from a NetCDF file and creates an animation
showing the evolution of the reconstructions over time for different camera ports.
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
    3: xr.open_dataset(ROOT / "data/sccm430/n=3_G32/tomograms/2025_10_06_23_28_03.nc"),
    4: xr.open_dataset(ROOT / "data/sccm430/n=4_G40/tomograms/2025_10_06_22_44_41.nc"),
    5: xr.open_dataset(ROOT / "data/sccm430/n=5_G53/tomograms/2025_10_06_23_43_24.nc"),
    7: xr.open_dataset(ROOT / "data/sccm430/n=7_G87/tomograms/2025_10_07_07_29_10.nc"),
}


# Assign coords (m -> mm)
for n_level, ds in tomograms.items():
    ds = ds.assign_coords(X=ds.X * 1e3, Y=ds.Y * 1e3)
    tomograms[n_level] = ds

# Set color limits
vlimits = {
    n_level: (
        ds.reconstructions.min(dim=["X", "Y", "tau"]),
        ds.reconstructions.max(dim=["X", "Y", "tau"]),
    )
    for n_level, ds in tomograms.items()
}

n_levels = list(tomograms.keys())

# %%
# Plot Emissivity - all ports, all n_levels
# =========================================
fig, axs = uplt.subplots(
    nrows=len(tomograms),
    ncols=tomograms[3].port.size,
    tight=False,  # NOTE: set tight=False to avoid changing layout during animation
    wspace=0,
    hspace=0.5,
    refwidth="3cm",
    dpi=100,
)

mappables = np.empty(
    (len(tomograms), tomograms[3].port.size),
    dtype=object,
)

# Plot for initial tau
for i_n_level, i_port in np.ndindex(len(tomograms), tomograms[3].port.size):
    ax = axs[i_n_level, i_port]
    n_level = n_levels[i_n_level]

    # Select data array
    da = tomograms[n_level]["reconstructions"].isel(tau=0, port=i_port)

    # Plot — Emissivity
    m = ax.pcolormesh(
        da.T,
        cmap="fire",
        discrete=False,
        vmin=vlimits[n_level][0].sel(port=da.port).item(),
        vmax=vlimits[n_level][1].sel(port=da.port).item(),
    )
    # ax.colorbar(
    #     m,
    #     tickdir="out",
    #     loc="ll",
    #     label="",
    # )

    mappables[i_n_level, i_port] = m

    ax.format(
        aspect="equal",
        ultitle=f"$n={n_level}$",
        urtitle=f"#{da.port.item()}",
    )

axs[0, 0].format(ltitle=f"{tomograms[3].tau.isel(tau=0):+.0f} µs")
axs.format(
    xlabel="$x$ [mm]",
    ylabel="$y$ [mm]",
    xlocator=20,
    ylocator=20,
    xminorlocator=5,
    yminorlocator=5,
    xlim=(-39.9, 39.9),
    ylim=(-39.9, 39.9),
)


def update(i_tau: int, mappables: np.ndarray, axs: np.ndarray) -> np.ndarray:
    """Update function for animation.

    Parameters
    ----------
    i_tau
        Time index to update the mappables for.
    mappables
        Current mappables for each n_level and port.
    axs
        Axes array corresponding to each n_level and port.

    Returns
    -------
    np.ndarray
        Updated mappables array.
    """
    for i_n_level, i_port in np.ndindex(len(tomograms), tomograms[3].port.size):
        # Update mappable
        mappables[i_n_level, i_port].set(
            array=tomograms[n_levels[i_n_level]]["reconstructions"]
            .isel(tau=i_tau, port=i_port)
            .T.data
        )

    # Update tau title
    axs[0, 0].format(
        ltitle=f"{tomograms[3].tau.isel(tau=i_tau):+.0f} µs",
    )

    return mappables


# Create animation
num_frames = tomograms[3].tau.size
ani = FuncAnimation(
    fig,
    update,
    fargs=(
        mappables,
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
        str(ROOT / "data/sccm430/tomograms_emissivity.mp4"),
        writer="ffmpeg",
        fps=60,
        progress_callback=lambda i, n: progress.update(task, advance=1),
    )

####################################################################################################

# %%
# Plot Emissivity Contours for windows #5
# =======================================

vmins_quantile = [
    ds.reconstructions.quantile(0.4, dim=["X", "Y", "tau"]) for ds in tomograms.values()
]

fig, axs = uplt.subplots(
    nrows=1,
    ncols=len(tomograms),
    tight=False,  # NOTE: set tight=False to avoid changing layout during animation
    refwidth="4.5cm",
)

mappables = []

# Plot for initial tau
for i, n_level in enumerate(n_levels):
    ax = axs[i]

    # Select data array
    da = tomograms[n_level]["reconstructions"].isel(tau=0, port=4)

    # Plot — Emissivity
    vmin = vmins_quantile[i].sel(port=da.port).item()
    vmax = vlimits[n_level][1].sel(port=da.port).item()
    m = ax.contour(
        da.T,
        cmap="fire",
        levels=15,
        vmin=vmin,
        vmax=vmax,
    )

    # Colorbar - Emissivity
    ax.colorbar(
        ScalarMappable(
            cmap="fire",
            norm=Normalize(
                vmin=vmin,
                vmax=vmax,
            ),
        ),
        label=f"Emissivity, $ɛ_{{{n_level}2}}$",
        loc="t",
        tickdir="out",
        pad=0.2,
        width=0.15,
    )

    ax.format(
        aspect="equal",
        ultitle=f"#{da.port.item()}\n$n={n_level}$",
        urtitle=f"{tomograms[3].tau.isel(tau=0):+.0f} µs",
    )

    mappables.append(m)


axs.format(
    xlabel="$x$ [mm]",
    ylabel="$y$ [mm]",
    xlocator=20,
    ylocator=20,
    xminorlocator=5,
    yminorlocator=5,
    xlim=(-39.9, 39.9),
    ylim=(-39.9, 39.9),
)


def update2(i_tau: int, mappables: list, axs: list) -> list:
    """Update function for animation.

    Update emissivity contours for each n_level at the given time index.

    Parameters
    ----------
    i_tau
        Time index to update the contours for.
    mappables
        List of current contour mappables for each n_level.
    axs
        List of axes corresponding to each n_level.

    Returns
    -------
    list
        Updated list of contour mappables.
    """
    for i, (n_level, mappable) in enumerate(zip(n_levels, mappables, strict=False)):
        # Select data array
        da = tomograms[n_level]["reconstructions"].isel(tau=i_tau, port=4)

        # Remove old contours
        mappable.remove()

        # Plot new contours
        contour = axs[i].contour(
            da.T,
            cmap="fire",
            levels=15,
            vmin=vmins_quantile[i].sel(port=da.port).item(),
            vmax=vlimits[n_level][1].sel(port=da.port).item(),
        )
        mappables[i] = contour

        axs[i].format(
            urtitle=f"{tomograms[3].tau.isel(tau=i_tau):+.0f} µs",
        )

    return mappables


# Create animation
num_frames = tomograms[3].tau.size
ani = FuncAnimation(
    fig,
    update2,
    fargs=(
        mappables,
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
        str(ROOT / "data/sccm430/tomograms_emissivity_contours.mp4"),
        writer="ffmpeg",
        fps=60,
        progress_callback=lambda i, n: progress.update(task, advance=1),
    )
