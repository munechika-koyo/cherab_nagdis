"""Module providing a set of functions to generate emission profiles."""

import numpy as np

__all__ = ["gauss", "complex_profile"]


def gauss(
    x: float,
    y: float,
    z: float,
    peak: float = 1.0,
    center: float = 0.0,
    deviation: float = 5.0e-3,
    limit: float = 10.0e-3,
) -> float:
    r"""Generate a Gaussian emission profile.

    The profile :math:`f(r)` is defined as:

    .. math::

        f(r) = A \left[
            \exp\left(
                -\frac{(r - r_\mathrm{center})^2}{\sigma^2}
            \right)
            - \exp\left(
                -\frac{(r_\mathrm{limit} - r_\mathrm{center})^2}{\sigma^2}
            \right)
        \right],

    where :math:`r = \sqrt{x^2 + y^2}` is the radial distance from the origin,
    :math:`A` is the peak value, :math:`r_\mathrm{center}` is the center position,
    :math:`\sigma` is the standard deviation, and :math:`r_\mathrm{limit}` is the limit position.

    Parameters
    ----------
    x
        X coordinate.
    y
        Y coordinate.
    z
        Z coordinate.
    peak
        Peak value of the Gaussian profile, :math:`A`.
    center
        Center position of the Gaussian profile, :math:`r_\mathrm{center}`.
    deviation
        Standard deviation of the Gaussian profile, :math:`\sigma`.
    limit
        Limit position for the Gaussian profile, :math:`r_\mathrm{limit}`.

    Returns
    -------
    float
        Value of the Gaussian profile at the given coordinates.
    """
    r = np.hypot(x, y)
    radiator = peak * (
        np.exp(-((r - center) ** 2) / deviation**2)
        - np.exp(-((limit - center) ** 2) / deviation**2)
    )

    return max(radiator, 0.0)


def complex_profile(
    x: float,
    y: float,
    z: float,
    peak: float = 1.0,
    radius_inner: float = 10.0e-3,
    radius_outer: float = 30.0e-3,
    dev_inner: float = 5.0e-3,
    dev_ring: float = 5.0e-3,
) -> float:
    r"""Generate a complex emission profile combining a central radiator and an outer ring radiator.

    The profile :math:`f(r, \theta)` is defined as:

    .. math::

        f(r, \theta) = f_\mathrm{central}(r) + f_\mathrm{ring}(r, \theta)

    where :math:`f_\mathrm{central}(r)` is the central radiator profile and
    :math:`f_\mathrm{ring}(r, \theta)` is the outer ring radiator profile.
    The central radiator profile is given by:

    .. math::

        f_\mathrm{central}(r) = A \left[
            \exp\left(
                -\frac{r^2}{\sigma_\mathrm{inner}^2}
            \right)
            - \exp\left(
                -\frac{r_\mathrm{inner}^2}{\sigma_\mathrm{inner}^2}
            \right)
        \right],

    and the outer ring radiator profile is given by:

    .. math::

        f_\mathrm{ring}(r, \theta) = A \cos(\theta) \exp\left(
            -\frac{(r - r_\mathrm{outer})^2}{\sigma_\mathrm{ring}^2}
        \right),

    where :math:`r = \sqrt{x^2 + y^2}` is the radial distance from the origin,
    :math:`\theta = \arctan2(y, x)` is the bearing
    :math:`A` is the peak value,
    :math:`r_\mathrm{inner}` and :math:`r_\mathrm{outer}` are the inner and outer radius,
    :math:`\sigma_\mathrm{inner}` and :math:`\sigma_\mathrm{ring}` are the standard deviations for
    the inner and ring radiators.

    Parameters
    ----------
    x
        X coordinate.
    y
        Y coordinate.
    z
        Z coordinate.
    peak
        Peak value of the emission profile, :math:`A`.
    radius_inner
        Inner radius for the central radiator, :math:`r_\mathrm{inner}`.
    radius_outer
        Outer radius for the ring radiator, :math:`r_\mathrm{outer}`.
    dev_inner
        Standard deviation for the central radiator, :math:`\sigma_\mathrm{inner}`.
    dev_ring
        Standard deviation for the ring radiator, :math:`\sigma_\mathrm{ring}`.

    Returns
    -------
    float
        Value of the complex emission profile at the given coordinates.
    """
    r = np.hypot(x, y)
    bearing = np.arctan2(y, x)

    central_radiatior = peak * (
        np.exp(-(r**2) / dev_inner**2) - np.exp(-(radius_inner**2) / dev_inner**2)
    )
    central_radiatior = max(0, central_radiatior)

    ring_radiator = peak * np.cos(bearing) * np.exp(-((r - radius_outer) ** 2) / dev_ring**2)
    ring_radiator = max(0, ring_radiator)

    return central_radiatior + ring_radiator
