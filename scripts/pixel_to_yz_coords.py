#!/usr/bin/env -S pixi run python
"""Convert pixel coordinates to y-z coordinates at x=0 plane using ray tracing."""

from pathlib import Path

import numpy as np
import xarray as xr
from calcam import Calibration
from raysect.optical import Point3D, World
from raysect.optical.loggingray import LoggingRay
from raysect.optical.material import AbsorbingSurface
from raysect.primitive.box import Box

from cherab.nagdis.tools.fetch import fetch_file

ROOT = Path(__file__).parents[1]

# %%
# Path definitions
# ----------------
path_to_ca = ROOT / "data" / "sccm430" / "n=3_G32" / "ca_s23_ma10.nc"
path_to_output = ROOT / "data" / "pixel_to_yz_coords.nc"

# %%
# Set up ray tracing environment
# ------------------------------
world = World()
calib = Calibration(fetch_file("20240705_mod.ccc"))
camera = calib.get_raysect_camera("Original")
camera.parent = world

# Add a y-z plane to terminate rays
Box(
    Point3D(-0.01, -100, -100),
    Point3D(0.0, 100, 100),
    material=AbsorbingSurface(),
    parent=world,
    name="Termination Plane",
)

# %%
# Prepare camera rays
# -------------------
# Trim the camera image to the active area only
# (This is specific to the 2024/7/5 camera setup)
h1 = 896
h2 = 192
h = (h1 - h2) // 2
trim = slice(h, h + h2)

origins = camera.pixel_origins[:, trim]
directions = camera.pixel_directions[:, trim]


def ray_paths(ix, iy):
    """Trace a ray from pixel (ix, iy) and return the origin and hit point."""
    ray = LoggingRay(origins[ix, iy], directions[ix, iy].normalise())
    ray.trace(world)

    return ray.path_vertices


# %%
# Load experimental dataset
# -------------------------
ds_ca = xr.open_dataset(path_to_ca)

ds = xr.Dataset(
    data_vars=dict(
        to_y=(
            ["port", "height", "width"],
            np.full((5, ds_ca.height.size, ds_ca.width.size), np.nan, dtype=float),
            dict(
                long_name="y coordinates",
                units="m",
                description="y coordinates at x=0 plane for each pixel",
            ),
        ),
        to_z=(
            ["port", "height", "width"],
            np.full((5, ds_ca.height.size, ds_ca.width.size), np.nan, dtype=float),
            dict(
                long_name="z coordinates",
                units="m",
                description="z coordinates at x=0 plane for each pixel",
            ),
        ),
    ),
    coords=dict(
        width=ds_ca.width,
        height=ds_ca.height,
        port=(
            ["port"],
            range(1, 6),
            dict(long_name="Port number", description="machine port number"),
        ),
    ),
)

for port in ds.port.data:
    pixel_rows, pixel_cols = np.where(ds_ca[f"mask-port-{port}"].data)

    for iy, ix in zip(pixel_rows, pixel_cols, strict=True):
        origin, hit_point = ray_paths(ix, iy)
        ds["to_y"][port - 1, iy, ix] = hit_point.z
        ds["to_z"][port - 1, iy, ix] = hit_point.y

ds = ds.assign(
    to_y_avg=(
        ["port", "height"],
        ds.to_y.mean(dim="width").data,
        dict(
            long_name="y-coordinate",
            units="m",
            description="y coordinate at x=0 plane, averaged along z-axis",
        ),
    ),
    to_z_avg=(
        ["port", "height"],
        ds.to_z.mean(dim="width").data,
        dict(
            long_name="z-coordinate",
            units="m",
            description="z coordinate at x=0 plane, averaged along y-axis",
        ),
    ),
)

# %%
# Save results
# ------------
ds.to_netcdf(path_to_output)
print(f"Saved to {path_to_output}")
