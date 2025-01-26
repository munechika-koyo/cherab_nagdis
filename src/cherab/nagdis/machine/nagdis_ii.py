"""Module to offer helper function to load plasma facing component meshes."""

from __future__ import annotations

from collections import defaultdict

from plotly import graph_objects as go
from plotly.graph_objects import Figure
from raysect.optical import World, rotate_z
from raysect.optical.material.absorber import AbsorbingSurface

# from raysect.optical.material.lambert import Lambert
from raysect.optical.material.material import Material
from raysect.primitive.mesh import Mesh
from rich.console import Console
from rich.progress import track
from rich.table import Table
from scipy.spatial.transform import Rotation

from ..tools.fetch import fetch_file
from .material import RoughSUS316L

__all__ = ["load_pfc_mesh", "show_PFCs_3D"]

# TODO: omtimization of roughness
SUS_ROUGHNESS = 0.0125

# List of Plasma Facing Components (filename is "**.rsm")
COMPONENTS: dict[str, tuple[str, Material, float | None]] = {
    # name: (filename, material class, roughness)
    "Vacuum Vessel Upper": ("vessel_upper", RoughSUS316L, SUS_ROUGHNESS),
    "Vacuum Vessel Lower": ("vessel_lower", RoughSUS316L, SUS_ROUGHNESS),
    "Gate Valve": ("gate_valve", RoughSUS316L, SUS_ROUGHNESS),
    # "Coils": ("coils", Lambert, None),
    # "Coils": ("coils_v2", Lambert, None),
}

# How many times each PFC element must be copy-pasted in toroidal direction
NCOPY: dict[str, int] = defaultdict(lambda: 1)

# Offset toroidal angle
ANG_OFFSET: dict[str, float] = defaultdict(lambda: 0.0)


def load_pfc_mesh(
    world: World,
    custom_materials: dict[str, tuple[Material, float | None]] | None = None,
    reflection: bool = True,
    is_fine_mesh: bool = True,
    show_result: bool = True,
    **kwargs,
) -> dict[str, list[Mesh]]:
    """Load plasma facing component meshes.

    Each mesh allows the user to use an user-defined material which inherites
    :obj:`~raysect.optical.material.material.Material`.

    Parameters
    ----------
    world : :obj:`~raysect.optical.world.World`
        The world scenegraph to which the meshes will be added.
    custom_materials : dict[str, tuple[Material, float | None]], optional
        User-defined material, by default None
        Set up like ``{"Vacuum Vessel Upper": (RoughSUS316L, 0.05), ...}``, where the key is the
        name of the mesh and the value is a tuple of material class and roughness.
    reflection : bool, optional
        Whether or not to consider reflection light, by default True.
        If ``False``, all of meshes' material are replaced to
        :obj:`~raysect.optical.material.absorber.AbsorbingSurface`.
    is_fine_mesh : bool, optional
        Whether or not to use fine mesh for the vacuum vessel, by default True.
    show_result : bool, optional
        Whether or not to show the result table of loading, by default True.
    **kwargs
        Keyword arguments to pass to `.fetch_file`.

    Returns
    -------
    dict[str, list[:obj:`~raysect.primitive.mesh.mesh.Mesh`]]
        Containing mesh name and :obj:`~raysect.primitive.mesh.mesh.Mesh` objects.

    Examples
    --------
    .. prompt:: python

        from raysect.optical import World
        from cherab.nagdis.machine import load_pfc_mesh

        world = World()
        meshes = load_pfc_mesh(world, reflection=True)
    """
    if is_fine_mesh:
        COMPONENTS["Vacuum Vessel Upper"] = ("vessel_upper_fine", RoughSUS316L, SUS_ROUGHNESS)
        COMPONENTS["Vacuum Vessel Lower"] = ("vessel_lower_fine", RoughSUS316L, SUS_ROUGHNESS)

    # Fetch meshes in advance
    paths_to_rsm = {}
    for mesh_name, (filename, _, _) in COMPONENTS.items():
        path = fetch_file(f"machine/{filename}.rsm", **kwargs)
        paths_to_rsm[mesh_name] = path

    meshes = {}
    statuses = []
    for mesh_name, (_, material_cls, roughness) in track(
        COMPONENTS.items(), description="Loading PFCs...", transient=True
    ):
        try:
            # Configure material
            if not reflection:
                material_cls = AbsorbingSurface
                roughness = None
            else:
                if custom_materials and mesh_name in custom_materials:
                    material_cls, roughness = custom_materials[mesh_name]
            if roughness is not None:
                material = material_cls(roughness=roughness)
            else:
                material = material_cls()

            # ================================
            # Load mesh
            # ================================
            # master element
            meshes[mesh_name] = [
                Mesh.from_file(
                    paths_to_rsm[mesh_name],
                    parent=world,
                    transform=rotate_z(ANG_OFFSET[mesh_name]),
                    material=material,
                    name=f"{mesh_name} 1" if NCOPY[mesh_name] > 1 else f"{mesh_name}",
                )
            ]
            # copies of the master element
            angle = 360.0 / NCOPY[mesh_name]
            for i in range(1, NCOPY[mesh_name]):
                meshes[mesh_name].append(
                    meshes[mesh_name][0].instance(
                        parent=world,
                        transform=rotate_z(angle * i + ANG_OFFSET[mesh_name]),
                        material=material,
                        name=f"{mesh_name} {i + 1}",
                    )
                )

            # Save the status of loading
            _status = "✅"
        except Exception as e:
            _status = f"❌ ({e})"
        finally:
            statuses.append(
                (
                    mesh_name,
                    paths_to_rsm[mesh_name],
                    material_cls.__name__,
                    str(roughness),
                    _status,
                )
            )

    if show_result:
        table = Table(title="Plasma Facing Components")
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Path to file", justify="left", style="magenta")
        table.add_column("Material", justify="center", style="green")
        table.add_column("Roughness", justify="center", style="yellow")
        table.add_column("Loaded", justify="center")
        for status in statuses:
            table.add_row(*status)
        console = Console()
        console.print(table)

    return meshes


def show_PFCs_3D(fig: Figure | None = None, fig_size: tuple[int, int] = (700, 500)) -> Figure:
    """Show Plasma Facing Components in 3-D space.

    Plot 3D meshes of PFCs with plotly.

    Parameters
    ----------
    fig : :obj:`~plotly.graph_objects.Figure`, optional
        Plotly Figure object, by default :obj:`~plotly.graph_objects.Figure`.
    fig_size : tuple[int, int], optional
        Figure size, by default (700, 500) pixel.

    Returns
    -------
    :obj:`~plotly.graph_objects.Figure`
        Plotly Figure object.

    Examples
    --------
    .. prompt:: python

        fig = show_PFCs_3D(fig_size=(700, 500))
        fig.show()

    The above codes automatically launch a browser to show the figure when it is executed in
    the python interpreter like the following picture:

    .. image:: ../_static/images/show_PFCs_3D_example.png
    """
    if fig is None or not isinstance(fig, Figure):
        fig = go.Figure()

    # load meshes
    world = World()
    meshes = load_pfc_mesh(world, reflection=False)

    for _, mesh_list in meshes.items():
        for mesh in mesh_list:
            # Rotate mesh by its transform matrix
            transform = mesh.to_root()
            r = Rotation.from_matrix([[transform[i, j] for j in range(3)] for i in range(3)])
            x, y, z = r.apply(mesh.data.vertices).T
            i, j, k = mesh.data.triangles.T

            # Create Mesh3d object
            mesh3D = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                flatshading=True,
                colorscale=[[0, "#e5dee5"], [1, "#e5dee5"]],
                intensity=z,
                name=f"{mesh.name}",
                text=f"{mesh.name}",
                showscale=False,
                showlegend=True,
                lighting=dict(
                    ambient=0.18,
                    diffuse=1,
                    fresnel=0.1,
                    specular=1,
                    roughness=0.1,
                    facenormalsepsilon=0,
                ),
                lightposition=dict(x=3000, y=3000, z=10000),
                hovertemplate=f"<b>{mesh.name}</b><br>" + "x: %{x}<br>y: %{y}<br>z: %{z}<br>"
                "<extra></extra>",
            )

            fig.add_trace(mesh3D)

    fig.update_layout(
        paper_bgcolor="rgb(1,1,1)",
        title_text="Device",
        title_x=0.5,
        font_color="white",
        hoverlabel_grouptitlefont_color="black",
        width=fig_size[0],
        height=fig_size[1],
        scene_aspectmode="data",
        margin=dict(r=10, l=10, b=10, t=35),
        scene_xaxis_visible=False,
        scene_yaxis_visible=False,
        scene_zaxis_visible=False,
    )

    return fig
