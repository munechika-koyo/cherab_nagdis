#!/usr/bin/env -S pixi run python
"""Generate a synthetic camera image with illumination to view the scene."""

from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint
from time import time

import numpy as np
import xarray as xr
from raysect.core.math import Point3D, translate
from raysect.optical import World
from raysect.optical.material import Lambert, UnitySurfaceEmitter
from raysect.optical.observer.pipeline import PowerPipeline2D
from raysect.primitive import Box

from cherab.nagdis.machine.nagdis_ii import load_pfc_mesh
from cherab.nagdis.observers import load_camera

ROOT = Path(__file__).resolve().parents[1]

# %%
# Create scene-graph
# ------------------
world = World()

# === load PFCs ===
reflection = True
meshes = load_pfc_mesh(
    world,
    custom_components={"Coils": ("coils_v2", Lambert, None)},
    reflection=reflection,
)

# === Environmental Emitters ===
Box(
    lower=Point3D(-100, -2e-2, -100),
    upper=Point3D(2, 2e-2, 100),
    material=UnitySurfaceEmitter(),
    transform=translate(0, 2.0, 0),
    parent=world,
)
Box(
    lower=Point3D(-1e-2, -100, -100),
    upper=Point3D(1e-2, 100, 100),
    material=UnitySurfaceEmitter(),
    transform=translate(-3.0, 0, 0),
    parent=world,
)


# %%
# Define observer pipeline
# ------------------------
power = PowerPipeline2D(name="Power", accumulate=False, display_progress=False)

# %%
# Load camera
# -----------
camera = load_camera(world)

# %%
# Set camera parameters
# ---------------------
camera.pipelines = [power]
# camera.pixels = (1280, 192)
camera.min_wavelength = 655.5
camera.max_wavelength = 657
camera.spectral_rays = 1
camera.spectral_bins = 1
camera.quiet = False
if hasattr(camera, "per_pixel_samples"):
    camera.per_pixel_samples = 50
    camera.lens_samples = 20


# %%
# Create xarray Dataset
# ---------------------
# xr.Dataset is used to store calculated data
ds = xr.Dataset()

# %%
# Save config
# -----------

# config info
config = {
    "camera_name": camera.name,
    "camera_pixels": camera.pixels,
    "camera_per_pixel_samples": camera.per_pixel_samples
    if hasattr(camera, "per_pixel_samples")
    else None,
    "camera_lens_samples": camera.lens_samples if hasattr(camera, "lens_samples") else None,
    "camera_spectral_bins": camera.spectral_bins,
    "camera_min_wavelength": camera.min_wavelength,
    "camera_max_wavelength": camera.max_wavelength,
    "PFCs": [mesh.name for _, mesh_list in meshes.items() for mesh in mesh_list],
    "reflection": int(reflection),
}
pprint(config)
ds = ds.assign_attrs(config)

# %%
# Render
# ------
start_time = time()

# render
camera.observe()

# Assign the synthetic powers to the dataset
ds = ds.assign(
    {
        "power": (
            ["x", "y"],
            power.frame.mean,
            dict(
                units="W",
                long_name="Synthetic power",
                description="Synthetic power received by each pixel",
            ),
        )
    },
)


ds = ds.assign_coords(
    width=(
        ["x"],
        np.arange(camera.pixels[0]),
        dict(units="pixel", long_name="Width", description="Pixel width"),
    ),
    height=(
        ["y"],
        np.arange(camera.pixels[1]),
        dict(units="pixel", long_name="Height", description="Pixel height"),
    ),
)


elapsed_time = timedelta(seconds=time() - start_time)
ds.attrs["elapsed_time"] = str(elapsed_time)

print(f"Elapsed time: {elapsed_time}")


# %%
# Save the dataset
# ----------------
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir = ROOT / "data" / "synthetic_images"
save_dir.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(save_dir / f"{time_now}.nc", format="NETCDF4")
print(f"Saved to {save_dir / f'{time_now}'}")
