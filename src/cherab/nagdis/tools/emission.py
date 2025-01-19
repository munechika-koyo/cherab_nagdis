"""This module provides a set of functions to generate emission profiles."""

import numpy as np

__all__ = ["gauss", "complex_profile"]


def gauss(x, y, z, peak=1.0, center=0.0, deviation=5.0e-3, limit=10.0e-3):
    r = np.hypot(x, y)
    radiator = peak * (
        np.exp(-((r - center) ** 2) / deviation**2)
        - np.exp(-((limit - center) ** 2) / deviation**2)
    )

    return max(radiator, 0.0)


def complex_profile(
    x,
    y,
    z,
    peak=1.0,
    radius_inner=10.0e-3,
    radius_outer=30.0e-3,
    dev_inner=5.0e-3,
    dev_ring=5.0e-3,
):
    r = np.hypot(x, y)
    bearing = np.arctan2(y, x)

    central_radiatior = peak * (
        np.exp(-(r**2) / dev_inner**2) - np.exp(-(radius_inner**2) / dev_inner**2)
    )
    central_radiatior = max(0, central_radiatior)

    ring_radiator = peak * np.cos(bearing) * np.exp(-((r - radius_outer) ** 2) / dev_ring**2)
    ring_radiator = max(0, ring_radiator)

    return central_radiatior + ring_radiator
