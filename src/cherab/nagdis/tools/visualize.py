"""Visualize module offers functions to visualize the data."""

from collections.abc import Callable, Collection
from typing import Literal, TypeAlias

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import XAxis, YAxis
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import ListedColormap, LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke
from matplotlib.ticker import (
    AutoLocator,
    AutoMinorLocator,
    EngFormatter,
    LogFormatterSciNotation,
    LogLocator,
    MultipleLocator,
    PercentFormatter,
    ScalarFormatter,
    SymmetricalLogLocator,
)
from mpl_toolkits.axes_grid1 import ImageGrid

from cherab.core.math import sample3d

__all__ = ["show_xy_profiles", "set_xy_axis_ticks", "set_axis_format", "set_norm"]


# Type aliases
PlotMode: TypeAlias = Literal["scalar", "log", "centered", "symlog", "asinh"]
FormatMode: TypeAlias = Literal["scalar", "log", "symlog", "asinh", "percent", "eng"]

# custom Red colormap extracted from "RdBu_r"
cmap = get_cmap("RdBu_r")
CMAP_RED = ListedColormap(list(cmap(np.linspace(0.5, 1.0, 256))))

# Constants
RADIUS_LIMIT = 35e-3


def show_xy_profiles(
    fig: Figure,
    func: Callable[[float, float, float], float],
    z: float = 0.0,
    resolution: float = 1e-3,
    xy_range: Collection[float] = (
        -RADIUS_LIMIT,
        RADIUS_LIMIT,
        -RADIUS_LIMIT,
        RADIUS_LIMIT,
    ),
    clabel: str = "",
    cmap: str = "RdBu_r",
    plot_mode: PlotMode = "centered",
) -> tuple[Figure, ImageGrid]:
    """Show 2D profile of a 3D function in the X-Y plane.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        Matplotlib Figure object to plot the profile.
    func : callable[[float, float, float], float]
        3D function to plot the profile in (X, Y, Z) cartesian coordinates.
    z : float, optional
        Z-coordinate of the X-Y plane, by default 0.0.
    resolution : float, optional
        Resolution of the profile, by default 1e-3.
    xy_range : Collection[float], optional
        Range of the XY plane, by default `(-35e-3, 35e-3, -35e-3, 35e-3)`.
    clabel : str, optional
        Colorbar label, by default "".
    cmap : str, optional
        Colormap name, by default `"RdBu_r"`.
    plot_mode : {`"scalar"`, `"log"`, `"centered"`, `"symlog"`, `"asinh"`}, optional
        Mode of the plot, by default `"centered"`.

    Returns
    -------
    tuple[`~matplotlib.figure.Figure`, `~mpl_toolkits.axes_grid1.ImageGrid`]
        Matplotlib Figure and ImageGrid objects.
    """
    if len(xy_range) != 4:
        raise ValueError("xy_range must be a 4-element collection.")
    xmin, xmax, ymin, ymax = xy_range
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid xy_range.")

    nx = round((xmax - xmin) / resolution)
    ny = round((ymax - ymin) / resolution)

    x_pts, y_pts, _, profile = sample3d(func, (xmin, xmax, nx), (ymin, ymax, ny), (z, z, 1))

    vmin, vmax = profile.min(), profile.max()

    norm = set_norm(plot_mode, vmin, vmax)

    # Plot
    grids = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 1),
        axes_pad=0.15,
        cbar_mode="single",
        cbar_location="right",
        cbar_size="5%",
        cbar_pad=0.0,
        share_all=True,
    )

    ax = grids[0]
    ax.pcolormesh(
        x_pts * 1.0e3,
        y_pts * 1.0e3,
        profile.squeeze().T,
        shading="auto",
        norm=norm,
        cmap=cmap,
    )
    ax.set_aspect("equal")

    set_xy_axis_ticks(ax)
    ax.set_xlabel("$X$ [mm]")
    ax.set_ylabel("$Y$ [mm]")

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(mappable, cax=grids.cbar_axes[0])
    set_axis_format(cbar.ax.yaxis, plot_mode)
    cbar.set_label(clabel)

    return fig, grids


def set_xy_axis_ticks(
    axes: Axes,
    x_interval: float | None = 1,
    y_interval: float | None = 1,
) -> None:
    """Set minor locators and ticks parameters of both x and y axes.

    Parameters
    ----------
    axes : `~matplotlib.axes.Axes`
        Matplotlib Axes object to set ticks.
    x_interval : float, optional
        Interval of x axis, by default 1 mm.
    y_interval : float, optional
        Interval of y axis, by default 1 mm.

    Examples
    --------

    .. prompt:: python

        import numpy as np
        from matplotlib import pyplot as plt
        from cherab.nagdis.tools.visualize import set_xy_axis_ticks

        x = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * x)

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(x, y)
        axes[1].plot(x, y)
        set_xy_axis_ticks(axes[1])
        plt.show()

    .. figure:: ../../_static/images/set_xy_axis_ticks.png
        :align: center

        The right figure is configured by `set_xy_axis_ticks` function.
    """
    if isinstance(x_interval, float):
        axes.xaxis.set_minor_locator(MultipleLocator(x_interval))
    if isinstance(y_interval, float):
        axes.yaxis.set_minor_locator(MultipleLocator(y_interval))
    axes.tick_params(direction="in", labelsize=10, which="both", top=True, right=True)


def set_axis_format(
    axis: XAxis | YAxis,
    formatter: FormatMode | PlotMode,
    linear_width: float = 1.0,
    offset_position: Literal["left", "right"] = "left",
    **kwargs,
) -> None:
    """Set axis format.

    Set specified axis major formatter and both corresponding major and minor locators.

    Parameters
    ----------
    axis : `~matplotlib.axis.XAxis` or `~matplotlib.axis.YAxis`
        Matplotlib axis object.
    formatter : {`"scalar"`, `"log"`, `"symlog"`, `"asinh"`, `"percent"`, `"eng"`}
        Formatter mode of the axis. Values in non-implemented modes are set to
        :obj:`~matplotlib.ticker.ScalarFormatter` with ``useMathText=True``.
    linear_width : float, optional
        Linear width of asinh/symlog norm, by default 1.0.
    offset_position : {"left", "right"}, optional
        Position of the offset text like :math:`\\times 10^3`, by default ``"left"``.
        This parameter only affects `~matplotlib.axis.YAxis` object.
    **kwargs
        Keyword arguments for formatter.
    """
    # define colobar formatter and locator
    if formatter == "log":
        fmt = LogFormatterSciNotation(**kwargs)
        major_locator = LogLocator(base=10, numticks=None)
        minor_locator = LogLocator(base=10, subs=tuple(np.arange(0.1, 1.0, 0.1)), numticks=12)

    elif formatter == "symlog":
        fmt = LogFormatterSciNotation(linthresh=linear_width, **kwargs)
        major_locator = SymmetricalLogLocator(linthresh=linear_width, base=10)
        minor_locator = SymmetricalLogLocator(
            linthresh=linear_width, base=10, subs=tuple(np.arange(0.1, 1.0, 0.1))
        )

    elif formatter == "asinh":
        raise NotImplementedError("asinh mode is not supported yet (due to old matplotlib).")

    elif formatter == "percent":
        fmt = PercentFormatter(**kwargs)
        major_locator = AutoLocator()
        minor_locator = AutoMinorLocator()

    elif formatter == "eng":
        fmt = EngFormatter(**kwargs)
        major_locator = AutoLocator()
        minor_locator = AutoMinorLocator()

    else:
        fmt = ScalarFormatter(useMathText=True, **kwargs)
        fmt.set_powerlimits((0, 0))
        major_locator = AutoLocator()
        minor_locator = AutoMinorLocator()

    # set axis properties
    if isinstance(axis, YAxis):
        axis.set_offset_position(offset_position)
    axis.set_major_formatter(fmt)
    axis.set_major_locator(major_locator)
    axis.set_minor_locator(minor_locator)


def set_norm(mode: PlotMode, vmin: float, vmax: float, linear_width: float = 1.0) -> Normalize:
    """Set specific :obj:`~matplotlib.colors.Normalize` object.

    Parameters
    ----------
    mode : {`"scalar"`, `"log"`, `"centered"`, `"symlog"`, `"asinh"`}
        The way of normalize the data scale.
    vmin : float
        Minimum value of the profile.
    vmax : float
        Maximum value of the profile.
    linear_width : float, optional
        Linear width of asinh/symlog norm, by default 1.0.

    Returns
    -------
    :obj:`~matplotlib.colors.Normalize`
        Normalize object corresponding to the mode.
    """
    # set norm
    absolute = max(abs(vmax), abs(vmin))
    if mode == "log":
        if vmin <= 0:
            raise ValueError("vmin must be positive value.")
        norm = LogNorm(vmin=vmin, vmax=vmax)

    elif mode == "symlog":
        norm = SymLogNorm(linthresh=linear_width, vmin=-1 * absolute, vmax=absolute)  # type: ignore

    elif mode == "centered":
        norm = Normalize(vmin=-1 * absolute, vmax=absolute)

    elif mode == "asinh":
        raise NotImplementedError("asinh norm is not supported yet (due to old matplotlib).")
        # norm = AsinhNorm(linear_width=linear_width, vmin=-1 * absolute, vmax=absolute)

    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


def _set_cbar_extend(user_vmin: float, user_vmax: float, data_vmin: float, data_vmax: float) -> str:
    """Set colorbar's extend.

    Parameters
    ----------
    user_vmin : float
        User defined minimum value.
    user_vmax : float
        User defined maximum value.
    data_vmin : float
        Minimum value of the profile.
    data_vmax : float
        Maximum value of the profile.

    Returns
    -------
    str
        Extend mode of the colorbar chosen from {"both", "min", "max", "neither"}.
    """
    if data_vmin < user_vmin:
        if user_vmax < data_vmax:
            extend = "both"
        else:
            extend = "min"
    else:
        if user_vmax < data_vmax:
            extend = "max"
        else:
            extend = "neither"

    return extend


def add_inner_title(
    ax: Axes,
    title: str,
    loc: str = "upper left",
    size: float = plt.rcParams["legend.fontsize"],
    borderpad: float = 0.5,
    **kwargs,
):
    """Add inner title to the axes.

    The text is padded by borderpad and has white stroke effect.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes object.
    title : str
        Title text.
    loc : str, optional
        Location of the title, by default "upper left".
    size : int, optional
        Font size of the title, by default `plt.rcParams["legend.fontsize"]`.
    borderpad : float, optional
        Padding of the title, by default 0.5.
    **kwargs
        Keyword arguments for `~matplotlib.offsetbox.AnchoredText`.

    Returns
    -------
    `~matplotlib.offsetbox.AnchoredText`
        AnchoredText object.
    """
    prop = dict(path_effects=[withStroke(linewidth=3, foreground="w")], size=size)
    at = AnchoredText(
        title, loc=loc, prop=prop, pad=0.0, borderpad=borderpad, frameon=False, **kwargs
    )
    ax.add_artist(at)
    return at
