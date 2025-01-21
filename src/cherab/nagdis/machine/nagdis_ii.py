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
from scipy.spatial.transform import Rotation

from cherab.inversion.tools import Spinner

from ..tools.fetch import fetch_file
from .material import RoughSUS316L

__all__ = ["load_pfc_mesh", "show_PFCs_3D"]

# TODO: omtimization of roughness
SUS_ROUGHNESS = 0.0125

# List of Plasma Facing Components (filename is "**.rsm")
COMPONENTS: dict[str, tuple[str, Material]] = {
    # name: (filename, material object)
    "Vacuum Vessel Upper": ("vessel_upper", RoughSUS316L(SUS_ROUGHNESS)),
    "Vacuum Vessel Lower": ("vessel_lower", RoughSUS316L(SUS_ROUGHNESS)),
    "Gate Valve": ("gate_valve", RoughSUS316L(SUS_ROUGHNESS)),
    # "Coils": ("coils", Lambert()),
    # "Coils": ("coils_v2", Lambert()),
}

# How many times each PFC element must be copy-pasted in toroidal direction
NCOPY: dict[str, int] = defaultdict(lambda: 1)

# Offset toroidal angle
ANG_OFFSET: dict[str, float] = defaultdict(lambda: 0.0)


def load_pfc_mesh(
    world: World,
    override_materials: dict[str, Material] | None = None,
    reflection: bool = True,
    is_fine_mesh: bool = True,
) -> dict[str, list[Mesh]]:
    """Load plasma facing component meshes.

    Each mesh allows the user to use an user-defined material which inherites
    :obj:`~raysect.optical.material.material.Material`.

    Parameters
    ----------
    world : :obj:`~raysect.optical.world.World`
        The world scenegraph belonging to these materials.
    override_materials : dict[str, Material], optional
        User-defined material. Set up like ``{"Vacuum Vessel Upper": RoughSUS316L(0.05), ...}``.
    reflection : bool, optional
        Whether or not to consider reflection light, by default True.
        If ``False``, all of meshes' material are replaced to
        :obj:`~raysect.optical.material.absorber.AbsorbingSurface`.
    is_fine_mesh : bool, optional
        Whether or not to use fine mesh for the vacuum vessel, by default True.

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
        COMPONENTS["Vacuum Vessel Upper"] = (
            "vessel_upper_fine",
            RoughSUS316L(SUS_ROUGHNESS),
        )
        COMPONENTS["Vacuum Vessel Lower"] = (
            "vessel_lower_fine",
            RoughSUS316L(SUS_ROUGHNESS),
        )

    meshes = {}

    with Spinner(text="Loading PFCs...") as spinner:
        for mesh_name, (filename, default_material) in COMPONENTS.items():
            try:
                spinner.text = f"Loading {mesh_name}..."

                # === fetch file ===
                path_to_rsm = fetch_file(f"machine/{filename}.rsm")

                # === set material ===
                if not reflection:
                    material = AbsorbingSurface()
                else:
                    if isinstance(override_materials, dict):
                        material = override_materials.get(mesh_name, None)
                        if material is None:
                            material = default_material
                        elif isinstance(material, Material):
                            pass
                        else:
                            raise TypeError(
                                f"override_materials[{mesh_name}] must be Material instance."
                            )
                    elif override_materials is None:
                        material = default_material
                    else:
                        raise TypeError(
                            f"override_materials must be dict[str, Material] instance or None. ({mesh_name})"
                        )

                # === load mesh ===
                # master element
                meshes[mesh_name] = [
                    Mesh.from_file(
                        path_to_rsm,
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

                # === print result ===
                material_str = str(material).split()[0].split(".")[-1]
                if roughness := getattr(material, "roughness", None):
                    material_str = f"{material_str: <12} (roughness: {roughness:.4f})"
                else:
                    material_str = f"{material_str}"
                spinner.write(f"✅ {mesh_name: <22}: {material_str}")

            except Exception as e:
                spinner.write(f"💥 {e}")

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
