"""Module to offer ray-transfer emitter objects."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from raysect.core.math import Point3D, Vector3D, rotate_vector, translate  # type: ignore
from raysect.core.scenegraph._nodebase import _NodeBase  # type: ignore

from cherab.tools.raytransfer import RayTransferBox, RayTransferCylinder

__all__ = [
    "create_raytransfer_cylinder",
    "create_raytransfer_box",
    "plot_rtc_grid",
    "plot_rtb_grid",
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
    parent : _NodeBase
        Parent node of the RayTransferCylinder object.
    radius : float, optional
        Radius of the cylinder, by default 40.0 mm.
    z_max : float, optional
        Maximum z-coordinate of the cylinder, by default 0.66 m.
    z_min : float, optional
        Minimum z-coordinate of the cylinder, by default -0.5 m.
    dr : float, optional
        Radial step size, by default 1.5 mm.
    dp : float, optional
        Polar step size, by default 2.0 degree.
    dz : float, optional
        Axial step size, by default 20 mm.
    step : float, optional
        Step size for the ray-transfer calculation, by default None.
        If None, the step size is set to 10% of the minimum of dr, dz, and dr * dp.

    Returns
    -------
    RayTransferCylinder
        RayTransferCylinder object.

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
    parent : _NodeBase
        Parent node of the RayTransferBox object.
    radius : float, optional
        Radius of the box, by default 40.0 mm.
    z_max : float, optional
        Maximum z-coordinate of the box, by default 0.66 m.
    z_min : float, optional
        Minimum z-coordinate of the box, by default -0.5 m.
    dx : float, optional
        Step size along the x-axis, by default 1.25 mm.
    dy : float, optional
        Step size along the y-axis, by default None.
        If None, dy is set to dx.
    dz : float, optional
        Step size along the z-axis, by default 20 mm.
    step : float, optional
        Step size for the ray-transfer calculation, by default None.
        If None, the step size is set to 10% of the minimum of dx, dy, and dz.

    Returns
    -------
    RayTransferBox
        RayTransferBox object.

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
    is_plot_axis=True,
    fig: Figure | None = None,
    axes: Axes | None = None,
    **kwargs,
) -> tuple[Figure, tuple[Axes, Axes] | Axes]:
    """Plot the grid of the RayTransferCylinder object.

    Parameters
    ----------
    rtc : RayTransferCylinder
        RayTransferCylinder object.
    is_plot_axis : bool, optional
        Whether to plot grids along the x and z axes, by default True.
    fig : Figure, optional
        Figure object to plot the grid, by default None.
    axes : Axes, optional
        Axes object to plot the grid, by default None.
    **kwargs
        Additional keyword arguments for the matplotlib plot function.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure object.
    axes1 : `~matplotlib.axes.Axes`
        Axes object for the cross-section grid.
    axes2 : `~matplotlib.axes.Axes`
        Axes object for the axial grid if `is_plot_axis` is True.
    """
    if not isinstance(rtc, RayTransferCylinder):
        raise TypeError("rtc must be a RayTransferCylinder object.")

    if is_plot_axis:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            dpi=200,
            sharey=True,
            gridspec_kw={"wspace": 0.01, "width_ratios": [1, 3]},
            figsize=(10, 3),
            layout="constrained",
        )
    else:
        if isinstance(fig, Figure):
            if axes is None:
                ax1 = fig.add_subplot(111)
            else:
                if not isinstance(axes, Axes):
                    raise TypeError("axes must be a matplotlib Axes object.")
                ax1 = axes
        else:
            fig, ax1 = plt.subplots(1, 1, dpi=200, layout="constrained")

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

        ax1.plot([start.x, end.x], [start.y, end.y], **kwargs)

    # Plot the circular lines
    for ir in range(nr + 1):
        radius = rmin + ir * dr
        angles = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        ax1.plot(x, y, **kwargs)

    ax1.set_xlabel("$X$ [m]")
    ax1.set_ylabel("$Y$ [m]")
    ax1.set_title("Cross-section grid")
    ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    # ============================================================================
    # Plot the axial grid in the x-z plane
    # ============================================================================
    if is_plot_axis:
        # Plot x-axis lines
        for iz in range(nz + 1):
            z = origin.z + iz * dz
            ax2.plot([z, z], [-rmax + origin.x, origin.x + rmax], **kwargs)

        # Plot z-axis lines
        for ir in range(2 * nr + 1):
            r = -rmax + ir * dr
            ax2.plot([origin.z, origin.z + height], [r, r], **kwargs)

        ax2.set_xlabel("$Z$ [m]")
        ax2.set_title("Axial grid")
        ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        return fig, (ax1, ax2)

    else:
        ax1.set_aspect("equal")
        return fig, ax1


def plot_rtb_grid(
    rtb: RayTransferBox,
    is_plot_axis=True,
    fig: Figure | None = None,
    axes: Axes | None = None,
    **kwargs,
) -> tuple[Figure, tuple[Axes, Axes] | Axes]:
    """Plot the grid of the RayTransferBox object.

    Parameters
    ----------
    rtb : RayTransferBox
        RayTransferBox object.
    is_plot_axis : bool, optional
        Whether to plot grids along the x and z axes, by default True.
    fig : Figure, optional
        Figure object to plot the grid, by default None.
    axes : Axes, optional
        Axes object to plot the grid, by default None.
    **kwargs
        Additional keyword arguments for the matplotlib plot function.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure object.
    axes1 : `~matplotlib.axes.Axes`
        Axes object for the cross-section grid.
    axes2 : `~matplotlib.axes.Axes`
        Axes object for the axial grid if `is_plot_axis` is True.
    """
    if not isinstance(rtb, RayTransferBox):
        raise TypeError("rtb must be a RayTransferBox object.")

    if is_plot_axis:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            dpi=200,
            sharey=True,
            gridspec_kw={"wspace": 0.01, "width_ratios": [1, 3]},
            figsize=(10, 3),
            layout="constrained",
        )
    else:
        if isinstance(fig, Figure):
            if axes is None:
                ax1 = fig.add_subplot(111)
            else:
                if not isinstance(axes, Axes):
                    raise TypeError("axes must be a matplotlib Axes object.")
                ax1 = axes
        else:
            fig, ax1 = plt.subplots(1, 1, dpi=200, layout="constrained")

    # Set plotting parameters
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linestyle", "-")
    kwargs.setdefault("linewidth", 0.5)

    # Get the values for the grid
    nx, ny, nz = rtb.material.grid_shape
    dx, dy, dz = rtb.material.grid_steps
    to_root = rtb._primitive.to_root()

    origin = ORIGIN.transform(to_root)
    basis_x = X_AXIS.transform(to_root)
    basis_y = Y_AXIS.transform(to_root)
    basis_z = Z_AXIS.transform(to_root)

    # ============================================================================
    # Plot the cross-section grid in the x-y plane
    # ============================================================================
    # Plot the x lines
    for iy in range(ny + 1):
        start = origin + (basis_y * iy * dy)
        end = start + (basis_x * nx * dx)

        ax1.plot([start.x, end.x], [start.y, end.y], **kwargs)

    # Plot the y lines
    for ix in range(nx + 1):
        start = origin + (basis_x * ix * dx)
        end = start + (basis_y * ny * dy)

        ax1.plot([start.x, end.x], [start.y, end.y], **kwargs)

    # Plot the limit circle line
    radius = nx * dx / 2
    thetas = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(
        radius * np.cos(thetas),
        radius * np.sin(thetas),
        **kwargs,
    )
    ax1.set_xlabel("$X$ [m]")
    ax1.set_ylabel("$Y$ [m]")
    ax1.set_title("Cross-section grid")
    ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    # ============================================================================
    # Plot the axial grid in the x-z plane
    # ============================================================================
    if is_plot_axis:
        # Plot x-axis lines
        for iz in range(nz + 1):
            start = origin + (basis_z * iz * dz)
            end = start + (basis_x * nx * dx)
            ax2.plot([start.z, end.z], [start.x, end.x], **kwargs)

        # Plot z-axis lines
        for ix in range(nx + 1):
            start = origin + (basis_x * ix * dx)
            end = start + (basis_z * nz * dz)
            ax2.plot([start.z, end.z], [start.x, end.x], **kwargs)

        ax2.set_xlabel("$Z$ [m]")
        ax2.set_title("Axial grid")
        ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        return fig, (ax1, ax2)

    else:
        ax1.set_aspect("equal")
        return fig, ax1


if __name__ == "__main__":
    from raysect.optical import World

    world = World()
    rtc = create_raytransfer_cylinder(world)
    rtb = create_raytransfer_box(world)
    fig, _ = plot_rtc_grid(rtc)
    fig, _ = plot_rtb_grid(rtb)
    plt.show()
