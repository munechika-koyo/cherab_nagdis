"""Utilities for the inversion functionality."""

import numpy as np
from raysect.core.math import Point3D, Vector3D, rotate_vector

from cherab.tools.raytransfer import (
    RayTransferBox,
    RayTransferCylinder,
    RayTransferObject,
)

__all__ = ["get_voxel_centers"]


ORIGIN = Point3D(0, 0, 0)
X_AXIS = Vector3D(1, 0, 0)
Y_AXIS = Vector3D(0, 1, 0)
Z_AXIS = Vector3D(0, 0, 1)


def get_voxel_centers(rto: RayTransferObject) -> np.ndarray:
    """Get the voxel center coordinates for a RayTransferObject.

    Parameters
    ----------
    rto : RayTransferObject
        The RayTransferObject for which to get the voxel centers.

    Returns
    -------
    ndarray
        The voxel center coordinates.
        If the `RayTransferObject` is a `RayTransferBox`, the shape of the array is `(nx, ny, nz, 3)`.
        If the `RayTransferObject` is a `RayTransferCylinder`, the shape of the array is `(nr, nphi, nz, 3)`.
    """
    if isinstance(rto, RayTransferObject):
        if isinstance(rto, RayTransferBox):
            is_cylindrical = False
            is_cartesian = True
        elif isinstance(rto, RayTransferCylinder):
            is_cylindrical = True
            is_cartesian = False
        else:
            raise NotImplementedError(f"RayTransferObject of type {type(rto)} not supported yet.")
    else:
        raise TypeError(f"{type(rto)} is not a supported RayTransferObject type.")

    # Get the rto origin and bases
    to_root = rto._primitive.to_root()
    origin = ORIGIN.transform(to_root)
    basis_x = X_AXIS.transform(to_root)
    basis_y = Y_AXIS.transform(to_root)
    basis_z = Z_AXIS.transform(to_root)

    if is_cartesian:
        dx, dy, dz = rto.material.grid_steps
        nx, ny, nz = rto.material.grid_shape

        voxel_centers = np.zeros((nx, ny, nz, 3))

        for i, j, k in np.ndindex(nx, ny, nz):
            center_point = (
                origin
                + (i + 0.5) * dx * basis_x
                + (j + 0.5) * dy * basis_y
                + (k + 0.5) * dz * basis_z
            )
            voxel_centers[i, j, k] = center_point.x, center_point.y, center_point.z

    elif is_cylindrical:
        dr, dphi, dz = rto.material.grid_steps
        nr, nphi, nz = rto.material.grid_shape

        voxel_centers = np.zeros((nr, nphi, nz, 3))

        for i, j in np.ndindex(nr, nphi):
            basis_r = basis_x.transform(rotate_vector((j + 0.5) * dphi, basis_z))

            for k in range(nz):
                center_point = origin + (i + 0.5) * dr * basis_r + (k + 0.5) * dz * basis_z
                voxel_centers[i, j, k] = center_point.x, center_point.y, center_point.z
    else:
        raise TypeError("Invalid RayTransferObject type.")

    return voxel_centers
