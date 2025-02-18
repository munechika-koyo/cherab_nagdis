"""This script is used to generate a Ray-Transfer Matrix (RTM)."""

from datetime import datetime, timedelta
from pathlib import Path
from time import time

import numpy as np
import xarray as xr
from raysect.core.math import Point3D, rotate_z
from raysect.optical import Node, World
from raysect.optical.observer import FullFrameSampler2D  # type: ignore
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from cherab.nagdis.inversion.raytransfer import (
    create_raytransfer_box,
    # create_raytransfer_cylinder,
)
from cherab.nagdis.inversion.utils import get_voxel_centers
from cherab.nagdis.machine.nagdis_ii import load_pfc_mesh
from cherab.nagdis.observers import load_camera
from cherab.tools.raytransfer import RayTransferBox, RayTransferCylinder, RayTransferPipeline2D

console = Console()

ROOT = Path(__file__).parents[1]


# %%
# Load sample dataset
# -------------------
ds_sample = xr.open_dataset(ROOT / "data" / "sccm430" / "n=3_G32" / "ca_s23.nc")

# %%
# Create Live progress
# --------------------
# group of progress bars;
# some are always visible, others will disappear when progress is complete
port_progress = Progress(
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
    console=console,
)
current_status = Progress(
    TextColumn("  "),
    TimeElapsedColumn(),
    TextColumn("[bold purple]{task.fields[action]}"),
    SpinnerColumn("simpleDots"),
    console=console,
)
angle_progress = Progress(
    TextColumn("[bold blue]Progress for angle {task.percentage:.0f}%"),
    BarColumn(),
    TextColumn("({task.completed} of {task.total} angle done)"),
    console=console,
)
# overall progress bar
overall_progress = Progress(
    TimeElapsedColumn(),
    BarColumn(),
    TextColumn("{task.description}"),
    console=console,
)
# Set progress panel
progress_panel = Group(
    Panel(
        Group(port_progress, current_status, angle_progress),
        title="RTM for Bolometers",
        title_align="left",
    ),
    overall_progress,
)

# %%
# Create scene-graph
# ------------------
world = World()

# load PFC meshes
meshes = load_pfc_mesh(world, reflection=False)

# load Ray-transfer object
node = Node(parent=world)
# ray_transfer = create_raytransfer_cylinder(node, dz=1.0)
ray_transfer = create_raytransfer_box(node, dz=1.0)

# Define observer pipeline
rtp = RayTransferPipeline2D(name="RayTransfer")

# Load camera
camera = load_camera(world, path_to_calibration="20240705_mod.ccc")

# Set camera parameters
camera.pipelines = [rtp]
camera.pixels = (1280, 192)
camera.min_wavelength = 655.5
camera.max_wavelength = 657
camera.spectral_rays = 1
camera.spectral_bins = ray_transfer.bins
camera.quiet = True
if hasattr(camera, "per_pixel_samples"):
    camera.per_pixel_samples = 50
    camera.lens_samples = 20


# %%
# Create xarray Dataset
# ---------------------
# xarray.Dataset is used to store ray-transfer-related data
ds = xr.Dataset()

# Set config info
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
    "RayTransfer_type": ray_transfer.__class__.__name__,
    "RayTransfer_bins": ray_transfer.bins,
    "RayTransfer_shape": ray_transfer.material.grid_shape,
    "RayTransfer_steps": ray_transfer.material.grid_steps,
    "RayTransfer_origin": tuple(Point3D(0, 0, 0).transform(ray_transfer._primitive.to_root())),
}
console.print(config)
ds = ds.assign_attrs(config)

# Save voxel info
grids = get_voxel_centers(ray_transfer)
if isinstance(ray_transfer, RayTransferBox):
    ds = ds.assign(
        voxel_centers=(
            ["X", "Y", "Z", "xyz"],
            grids,
            dict(units="m", long_name="Voxel centers"),
        ),
        voxel_map=(
            ["X", "Y", "Z"],
            ray_transfer.voxel_map,
            dict(units="pixel", long_name="Voxel map"),
        ),
        voxel_mask=(
            ["X", "Y", "Z"],
            ray_transfer.mask,
            dict(units="bool", long_name="Voxel mask"),
        ),
    )
    ds = ds.assign_coords(
        X=(
            ["X"],
            grids[:, 0, 0, 0].ravel(),
            dict(units="m", description="Horizontal axis"),
        ),
        Y=(
            ["Y"],
            grids[0, :, 0, 1].ravel(),
            dict(units="m", description="Vertical axis"),
        ),
        Z=(
            ["Z"],
            grids[0, 0, :, 2],
            dict(units="m", description="Magnetic axis"),
        ),
        xyz=(
            ["xyz"],
            ["x", "y", "z"],
            dict(description="Cartesian coordinates"),
        ),
    )
elif isinstance(ray_transfer, RayTransferCylinder):
    ds = ds.assign(
        voxel_centers=(
            ["R", "p", "Z", "xyz"],
            grids,
            dict(units="m", long_name="Voxel centers"),
        ),
        voxel_map=(
            ["R", "p", "Z"],
            ray_transfer.voxel_map,
            dict(units="pixel", long_name="Voxel map"),
        ),
        voxel_mask=(
            ["R", "p", "Z"],
            ray_transfer.mask,
            dict(units="bool", long_name="Voxel mask"),
        ),
    )
    ds = ds.assign_coords(
        R=(
            ["R"],
            np.hypot(grids[:, 0, 0, 0], grids[:, 0, 0, 0]).ravel(),
            dict(units="m", description="Major Radius"),
        ),
        p=(
            ["p"],
            np.arctan2(grids[0, :, 0, 1], grids[0, :, 0, 0]).ravel(),
            dict(units="rad", description="Polar angle"),
        ),
        Z=(
            ["Z"],
            grids[0, 0, :, 2],
            dict(units="m", description="Magnetic axis"),
        ),
        xyz=(
            ["xyz"],
            ["x", "y", "z"],
            dict(description="Cartesian coordinates"),
        ),
    )
else:
    raise NotImplementedError(f"RayTransferObject of type {type(ray_transfer)} not supported yet.")


# %%
# Calculate the ray-transfer matrix
# ---------------------------------
start_time = time()

# Set the iteration variables
ports = [1, 2, 3, 4, 5]
rotate_time = 70e-6  # 70μs
camera_interval = (ds_sample["tau"][1] - ds_sample["tau"][0]).item()  # [s]
num_angles = round(rotate_time / camera_interval)
angles = np.linspace(-90, 90, num_angles // 2, endpoint=True)

# Set overall progress task
overall_task_id = overall_progress.add_task("", total=len(ports))

with Live(progress_panel):
    # changing the angle of the cylinder and port masking
    for i_port, port in enumerate(ports):
        # Set overall task
        overall_progress.update(
            overall_task_id,
            description=f"[bold #AAAAAA]({i_port} out of {len(ports)} port done)",
        )

        # Initialize current task
        port_task_id = port_progress.add_task(f"{port}")
        current_task_id = current_status.add_task("", action="Preprocessing")
        angle_task_id = angle_progress.add_task("", total=len(angles))

        # Set frame sampler
        camera_mask = ds_sample[f"mask-port-{port}"].values
        camera.frame_sampler = FullFrameSampler2D(mask=camera_mask.T.copy(order="C"))

        # temporary storage for the ray-transfer matrix
        rtm = np.zeros((len(angles), camera_mask.sum(), ray_transfer.bins))

        # %%
        # Rotate the Raytransfer Emitter
        # ------------------------------
        for i_angle, angle in enumerate(angles):
            # Update the current task
            current_status.update(current_task_id, action=f"Rendering Angle {angle:.1f}°")

            # rotate the cylinder
            node.transform = rotate_z(angle)

            # render
            camera.observe()

            # flip the matrix to match the camera mask
            _rtm = np.transpose(rtp.matrix, (1, 0, 2))  # type: ignore  (x, y, bins) -> (y, x, bins)

            # Assign the ray-transfer matrix to the temporary storage
            rtm[i_angle] = _rtm[camera_mask, :] / (4.0 * np.pi)  # [m^3 sr] -> [m^3]

            # Advance the angle task
            angle_progress.advance(angle_task_id)

        # Unvisible the angle task
        angle_progress.update(angle_task_id, visible=False)

        current_status.update(current_task_id, action="Postprocessing")

        # Save rtm to the dataset
        ds = ds.assign(
            {
                f"rtm_port_{port}": (
                    ["angle", f"px_{port}", "bins"],
                    rtm,
                    dict(units="m^3", long_name=f"Ray-transfer matrix port-{port} for each angle"),
                )
            }
        )
        # Save the camera mask to the dataset
        ds = ds.assign(
            {
                f"camera_mask_port_{port}": (
                    ["y", "x"],
                    camera_mask,
                    dict(
                        long_name=f"Camera Pixel Mask Port-{port}",
                        description="element is True where the pixel is calculated",
                    ),
                )
            },
        )
        # Assign the pixel coordinates to the dataset
        ds = ds.assign_coords(
            {
                f"pixels_port_{port}": (
                    [f"px_{port}"],
                    np.arange(camera_mask.sum()),
                    dict(long_name="Pixel", description="Pixel index"),
                )
            }
        )

        # Finalize the port task
        port_progress.stop_task(port_task_id)
        port_progress.update(port_task_id, description=f"[bold green]Port {port} done!")

    ds = ds.assign_coords(
        angle=(
            ["angle"],
            angles,
            dict(units="deg", long_name="Angle", description="Rotation angle of the cylinder"),
        ),
        width=(
            ["x"],
            np.arange(camera_mask.shape[1]),
            dict(units="pixel", long_name="Width", description="Pixel width"),
        ),
        height=(
            ["y"],
            np.arange(camera_mask.shape[0]),
            dict(units="pixel", long_name="Height", description="Pixel height"),
        ),
        bins=(
            ["bins"],
            np.arange(ray_transfer.bins),
            dict(long_name="Voxel index", description="Spectral bin (voxel) index"),
        ),
    )

    # Finalize progress
    overall_progress.update(
        overall_task_id,
        description="[bold green]All ports done!",
    )

elapsed_time = timedelta(seconds=time() - start_time)
ds.attrs["elapsed_time"] = str(elapsed_time)

# %%
# Save the dataset
# ----------------
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir = ROOT / "data" / "rtms"
save_dir.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(save_dir / f"{time_now}.nc")
console.print(f"[bold green]Saved to[bold green] {save_dir / f'{time_now}'}")
