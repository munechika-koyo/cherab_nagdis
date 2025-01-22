"""Module to offer helper function to load fast camera."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from calcam.calibration import Calibration
from raysect.core.math import (
    AffineMatrix3D,
    Point3D,
    Vector3D,
    extract_translation,
    rotate_x,
)
from raysect.optical import World
from raysect.optical.observer import Observer2D

from ..tools.fetch import fetch_file
from .thin_lens_ccd import ThinLensCCDArray

__all__ = ["load_camera", "show_camera_geometry"]


def load_camera(
    parent: World,
    path_to_calibration: str = "20240705_mod.ccc",
    **kwargs,
) -> ThinLensCCDArray:
    """Loading fast lens camera configured with calcam calibration data.

    Default camera extrinsic matrix (rotation matrix and translation vector) is loaded from
    `calcam` calibration data.

    Parameters
    ----------
    parent : `~raysect.optical.scenegraph.world.World`
        Raysect world object to which the camera is attached.
    path_to_calibration : str, optional
        Path to `calcam` calibration data, by default "20240705_mod.ccc".
        This file is fetched by `.fetch_file` function.
    **kwargs
        Additional keyword arguments to pass to `.fetch_file` function.

    Returns
    -------
    `.ThinLensCCDArray`
        Instance of ThinLensCCDArray object.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.optical import World
        >>> from cherab.nagdis.observers import load_camera
        >>>
        >>> world = World()
        >>> camera = load_camera(world)
    """
    try:
        # Load calibration data from calcam file
        path = fetch_file(path_to_calibration, **kwargs)
        calib = Calibration(path)

        # Get camera matrix, rotation matrix and translation vector
        camera_matrix = calib.get_cam_matrix()
        camera_pos = calib.get_pupilpos(coords="Original")
        rotation_matrix = calib.get_cam_to_lab_rotation()

        # Set camera extrinsic matrix
        transform = AffineMatrix3D(
            np.block(
                [
                    [rotation_matrix, camera_pos.reshape(3, 1)],
                    [np.array([0, 0, 0, 1])],
                ]
            )
        )

        # === generate ThinLensCCDArray object ===
        pixel_size = calib.pixel_size
        camera = ThinLensCCDArray(
            pixels=(1280, 896),
            width=pixel_size * 1280,
            focal_length=35e-3,
            # working_distance=1.7,
            ccd_distance=pixel_size * camera_matrix[0, 0],
            f_number=12,
            parent=parent,
            pipelines=None,
            transform=rotate_x(-90)
            * transform,  # NOTE: rotate_x(-90) is mondatory for +Y up system
            name="Fast-visible camera",
        )
    except Exception as e:
        raise e

    return camera


def show_camera_geometry(fig: go.Figure, camera: Observer2D) -> go.Figure:
    """Show camera geometry.

    Parameters
    ----------
    fig : `~plotly.graph_objects.Figure`
        Plotly figure object.
    camera : `~raysect.optical.observer.Observer2D`
        Observer2D object.

    Returns
    -------
    `~plotly.graph_objects.Figure`
        Plotly figure object with camera geometry.
    """
    to_root = camera.to_root()

    # Camera position
    camera_pos = Point3D(*extract_translation(to_root))

    # Camera's x, y, z axis
    basis_x = to_root * Vector3D(1, 0, 0)
    basis_y = to_root * Vector3D(0, 1, 0)
    basis_z = to_root * Vector3D(0, 0, 1)

    width = 30e-2

    # Plot camera's axis
    xaxis_vector = go.Scatter3d(
        x=[camera_pos.x, camera_pos.x + width * basis_x.x],
        y=[camera_pos.y, camera_pos.y + width * basis_x.y],
        z=[camera_pos.z, camera_pos.z + width * basis_x.z],
        marker=dict(color="rgb(256, 0, 0)", size=2),
        line=dict(color="rgb(256, 0, 0)"),
        showlegend=False,
    )

    yaxis_vector = go.Scatter3d(
        x=[camera_pos.x, camera_pos.x + width * basis_y.x],
        y=[camera_pos.y, camera_pos.y + width * basis_y.y],
        z=[camera_pos.z, camera_pos.z + width * basis_y.z],
        marker=dict(color="rgb(0, 256, 0)", size=2),
        line=dict(color="rgb(0, 256, 0)"),
        showlegend=False,
    )

    zaxis_vector = go.Scatter3d(
        x=[camera_pos.x, camera_pos.x + width * basis_z.x],
        y=[camera_pos.y, camera_pos.y + width * basis_z.y],
        z=[camera_pos.z, camera_pos.z + width * basis_z.z],
        marker=dict(color="rgb(0, 0, 256)", size=2),
        line=dict(color="rgb(0, 0, 256)"),
        showlegend=False,
    )

    fig.add_trace(xaxis_vector)
    fig.add_trace(yaxis_vector)
    fig.add_trace(zaxis_vector)

    return fig
