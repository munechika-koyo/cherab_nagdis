"""Module to offer ray-transfer emitter objects."""

from __future__ import annotations

import numpy as np
import ultraplot as uplt
from raysect.core.math import Point3D, Vector3D, rotate_vector, translate
from raysect.core.scenegraph._nodebase import _NodeBase

from cherab.tools.raytransfer import RayTransferBox, RayTransferCylinder

__all__ = [
    "create_raytransfer_cylinder",
    "create_raytransfer_box",
    "plot_rtc_grid",
    "plot_rtb_grid",
    "plot_rtb_cross_section_grid",
]

# Constants
INSIDE_RADIUS = 0.088
ORIGIN = Point3D(0, 0, 0)
X_AXIS = Vector3D(1, 0, 0)
Y_AXIS = Vector3D(0, 1, 0)
Z_AXIS = Vector3D(0, 0, 1)


def create_raytransfer_cylinder(
    parent: _NodeBase,
    radius: float = 40.0e-3,
    z_max: float = 0.66,
    z_min: float = -0.5,
    dr: float = 1.5e-3,
    dp: float = 2.0,
    dz: float = 20e-3,
    step: float | None = None,
) -> RayTransferCylinder:
    """Create a RayTransferCylinder object with the given parameters.

    The z axis is aligned with the linear device magnetic axis.

    Parameters
    ----------
    parent
        Parent node of the RayTransferCylinder object.
    radius
        Radius of the cylinder, by default 40.0 mm.
    z_max
        Maximum z-coordinate of the cylinder, by default 0.66 m.
    z_min
        Minimum z-coordinate of the cylinder, by default -0.5 m.
    dr
        Radial step size, by default 1.5 mm.
    dp
        Polar step size, by default 2.0 degree.
    dz
        Axial step size, by default 20 mm.
    step
        Step size for the ray-transfer calculation, by default None.
        If None, the step size is set to 10% of the minimum of dr, dz, and dr * dp.

    Returns
    -------
    RayTransferCylinder
        RayTransferCylinder object.

    Raises
    ------
    TypeError
        If `parent` is not a scene-graph object.
    ValueError
        If `radius`, `z_max`, `z_min`, `dr`, `dp`, `dz`, or `step` are invalid.

    Examples
    --------
    >>> from raysect.optical import World
    >>> from cherab.nagdis.inversion.raytransfer import create_raytransfer_cylinder
    >>> world = World()
    >>> cylinder = create_raytransfer_cylinder(world)
    """
    if not isinstance(parent, _NodeBase):
        raise TypeError("Parent must be a scene-graph object.")

    if radius <= 0 or radius > INSIDE_RADIUS:
        raise ValueError(f"Radius must be in (0, {INSIDE_RADIUS}). {radius=}")

    if z_max <= z_min:
        raise ValueError(f"z_max must be greater than z_min ({z_max=}, {z_min=}).")

    if dr <= 0 or dp <= 0 or dz <= 0:
        raise ValueError(f"dr, dp, dz must be positive numbers. (dr, dp, dz) = ({dr}, {dp}, {dz})")

    height = z_max - z_min
    n_radius = round(radius / dr)
    n_polar = round(360 / dp)
    n_height = round(height / dz)

    if step is not None:
        if step <= 0:
            raise ValueError(f"step must be a positive number. {step=}")
    else:
        step = 0.1 * min(dr, dz, dr * np.deg2rad(dp))

    return RayTransferCylinder(
        radius_outer=radius,
        radius_inner=0,
        height=height,
        n_radius=n_radius,
        n_height=n_height,
        n_polar=n_polar,
        step=step,
        parent=parent,
        transform=translate(0, 0, z_min),
    )


def create_raytransfer_box(
    parent: _NodeBase,
    radius: float = 40.0e-3,
    z_max: float = 0.66,
    z_min: float = -0.5,
    dx: float = 1.25e-3,
    dy: float | None = None,
    dz: float = 20e-3,
    step: float | None = None,
) -> RayTransferBox:
    """Create a RayTransferBox object with the given parameters.

    The z axis is aligned with the linear device magnetic axis.

    .. note::

        We need to avoid having the center of the voxel coincide with the origin.
        This is because when we set the gradient-based derivative matrices,
        the origin point will be singular.

    Parameters
    ----------
    parent
        Parent node of the RayTransferBox object.
    radius
        Radius of the box, by default 40.0 mm.
    z_max
        Maximum z-coordinate of the box, by default 0.66 m.
    z_min
        Minimum z-coordinate of the box, by default -0.5 m.
    dx
        Step size along the x-axis, by default 1.25 mm.
    dy
        Step size along the y-axis, by default None.
        If None, dy is set to dx.
    dz
        Step size along the z-axis, by default 20 mm.
    step
        Step size for the ray-transfer calculation, by default None.
        If None, the step size is set to 10% of the minimum of dx, dy, and dz.

    Returns
    -------
    RayTransferBox
        RayTransferBox object.

    Raises
    ------
    TypeError
        If `parent` is not a scene-graph object.
    ValueError
        If `radius`, `z_max`, `z_min`, `dx`, `dy`, `dz`, or `step` are invalid.

    Examples
    --------
    >>> from raysect.optical import World
    >>> from cherab.nagdis.inversion.raytransfer import create_raytransfer_box
    >>> world = World()
    >>> box = create_raytransfer_box(world)
    """
    if not isinstance(parent, _NodeBase):
        raise TypeError("Parent must be a scene-graph object.")

    if radius <= 0 or radius > INSIDE_RADIUS:
        raise ValueError(f"Radius must be in (0, {INSIDE_RADIUS}). {radius=}")

    if z_max <= z_min:
        raise ValueError(f"z_max must be greater than z_min ({z_max=}, {z_min=}).")

    if dy is None:
        dy = dx

    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError(f"dr, dy, dz must be positive numbers. (dr, dy, dz) = ({dx}, {dy}, {dz})")

    height = z_max - z_min
    n_x = round(radius * 2 / dx)
    n_y = round(radius * 2 / dy)
    n_height = round(height / dz)

    if step is not None:
        if step <= 0:
            raise ValueError(f"step must be a positive number. {step=}")

    # Create a cylindrical mask for the box
    x = np.linspace(-radius + 0.5 * dx, radius - 0.5 * dx, n_x, endpoint=True)
    xsqrt = x * x
    mask = xsqrt[:, None, None] + xsqrt[None, :, None] <= radius * radius
    mask = np.repeat(mask[:, :], n_height, axis=2)

    return RayTransferBox(
        xmax=radius * 2,
        ymax=radius * 2,
        zmax=height,
        nx=n_x,
        ny=n_y,
        nz=n_height,
        step=step,
        mask=mask,
        parent=parent,
        transform=translate(-radius, -radius, z_min),
    )


def plot_rtc_grid(
    rtc: RayTransferCylinder,
    is_plot_axis: bool = True,
    **kwargs,
) -> tuple[uplt.Figure, uplt.gridspec.GridSpec]:
    """Plot the grid of the RayTransferCylinder object.

    Parameters
    ----------
    rtc
        RayTransferCylinder object.
    is_plot_axis
        Whether to plot grids along the x and z axes, by default True.
    **kwargs
        Additional keyword arguments for the matplotlib plot function.

    Returns
    -------
    fig : `~ultraplot.figure.Figure`
        Figure object.
    axs: `~ultraplot.gridspec.SubplotGrid`
        Axes object for the cross-section grid and axial grid if `is_plot_axis` is True.

    Raises
    ------
    TypeError
        If `rtc` is not a RayTransferCylinder object.
    """
    if not isinstance(rtc, RayTransferCylinder):
        raise TypeError("rtc must be a RayTransferCylinder object.")

    if is_plot_axis:
        fig, axs = uplt.subplots(
            nrows=1,
            ncols=2,
            sharex=False,
            sharey=True,
        )
    else:
        fig, axs = uplt.subplots()

    # Set plotting parameters
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("linewidth", 0.5)

    # Get the values for the grid
    nr, ntheta, nz = rtc.material.grid_shape
    dr, dp, dz = rtc.material.grid_steps
    to_root = rtc._primitive.to_root()
    rmin = rtc.material.rmin
    rmax = rmin + nr * dr
    height = nz * dz

    origin = ORIGIN.transform(to_root)
    basis_x = X_AXIS.transform(to_root)
    basis_z = Z_AXIS.transform(to_root)

    # ============================================================================
    # Plot the cross-section grid in the x-y plane
    # ============================================================================
    # Plot the radial lines
    for ip in range(ntheta):
        start = origin
        end = origin + (basis_x * rmax).transform(rotate_vector(ip * dp, basis_z))

        axs[0].plot([start.x, end.x], [start.y, end.y], **kwargs)

    # Plot the circular lines
    for ir in range(nr + 1):
        radius = rmin + ir * dr
        angles = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        axs[0].plot(x, y, **kwargs)

    axs[0].format(
        aspect="equal",
        xlabel="$X$ [m]",
        ylabel="$Y$ [m]",
        title="Cross-section grid",
    )

    # ============================================================================
    # Plot the axial grid in the x-z plane
    # ============================================================================
    if is_plot_axis:
        # Plot x-axis lines
        for i in range(nz + 1):
            z = origin.z + i * dz
            axs[1].plot([z, z], [-rmax + origin.x, origin.x + rmax], **kwargs)

        # Plot z-axis lines
        for ir in range(2 * nr + 1):
            r = -rmax + ir * dr
            axs[1].plot([origin.z, origin.z + height], [r, r], **kwargs)

        axs[1].format(
            xlabel="$Z$ [m]",
            title="Axial grid",
        )

    axs.format(
        xreverse=False,
    )

    return fig, axs


def plot_rtb_grid(
    rtb: RayTransferBox,
    is_plot_axis: bool = True,
    **kwargs,
) -> tuple[uplt.Figure, uplt.gridspec.SubplotGrid]:
    """Plot the grid of the RayTransferBox object.

    Parameters
    ----------
    rtb
        RayTransferBox object.
    is_plot_axis
        Whether to plot grids along the x and z axes, by default True.
    **kwargs
        Additional keyword arguments for the matplotlib plot function.

    Returns
    -------
    fig : `~ultraplot.figure.Figure`
        Figure object.
    axs: `~ultraplot.gridspec.SubplotGrid`
        Subplot grid object.

    Raises
    ------
    TypeError
        If `rtb` is not a RayTransferBox object.
    """
    if not isinstance(rtb, RayTransferBox):
        raise TypeError("rtb must be a RayTransferBox object.")

    if is_plot_axis:
        fig, axs = uplt.subplots(
            nrows=1,
            ncols=2,
            sharex=False,
            sharey=True,
        )
    else:
        fig, axs = uplt.subplots()

    # Set plotting parameters
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("linewidth", 0.5)

    ax = plot_rtb_cross_section_grid(rtb, axs[0], **kwargs)

    # Get the values for the grid
    nx, ny, nz = rtb.material.grid_shape
    dx, dy, dz = rtb.material.grid_steps
    to_root = rtb._primitive.to_root()

    origin = ORIGIN.transform(to_root)
    basis_x = X_AXIS.transform(to_root)
    basis_z = Z_AXIS.transform(to_root)

    # Plot the limit circle line
    radius = nx * dx / 2
    thetas = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        radius * np.cos(thetas),
        radius * np.sin(thetas),
        **kwargs,
    )
    ax.format(
        aspect="equal",
        xlabel="$X$ [m]",
        ylabel="$Y$ [m]",
        title="Cross-section grid",
    )

    # ============================================================================
    # Plot the axial grid in the x-z plane
    # ============================================================================
    if is_plot_axis:
        # Plot x-axis lines
        for i in range(nz + 1):
            start = origin + (basis_z * i * dz)
            end = start + (basis_x * nx * dx)
            axs[1].plot([start.z, end.z], [start.x, end.x], **kwargs)

        # Plot z-axis lines
        for ix in range(nx + 1):
            start = origin + (basis_x * ix * dx)
            end = start + (basis_z * nz * dz)
            axs[1].plot([start.z, end.z], [start.x, end.x], **kwargs)

        axs[1].format(
            xlabel="$Z$ [m]",
            title="Axial grid",
        )

    axs.format(
        xreverse=False,
    )

    return fig, axs


def plot_rtb_cross_section_grid(
    rtb: RayTransferBox,
    ax,
    **kwargs,
) -> uplt.axes.Axes:
    """Plot only the x-y cross-section grid lines of a RayTransferBox on a given axis.

    Parameters
    ----------
    rtb
        RayTransferBox object.
    ax
        Axis object to draw the cross-section grid onto.
    **kwargs
        Additional keyword arguments for the matplotlib plot function.

    Returns
    -------
    `~ultraplot.axes.Axes`
        Axis object with cross-section grid lines.

    Raises
    ------
    TypeError
        If `rtb` is not a RayTransferBox object.
    """
    if not isinstance(rtb, RayTransferBox):
        raise TypeError("rtb must be a RayTransferBox object.")

    # Set default plotting parameters
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("linewidth", 0.5)

    nx, ny, _ = rtb.material.grid_shape
    dx, dy, _ = rtb.material.grid_steps
    to_root = rtb._primitive.to_root()

    origin = ORIGIN.transform(to_root)
    basis_x = X_AXIS.transform(to_root)
    basis_y = Y_AXIS.transform(to_root)

    # Plot the x lines
    for iy in range(ny + 1):
        start = origin + (basis_y * iy * dy)
        end = start + (basis_x * nx * dx)
        ax.plot([start.x, end.x], [start.y, end.y], **kwargs)

    # Plot the y lines
    for ix in range(nx + 1):
        start = origin + (basis_x * ix * dx)
        end = start + (basis_y * ny * dy)
        ax.plot([start.x, end.x], [start.y, end.y], **kwargs)

    return ax
