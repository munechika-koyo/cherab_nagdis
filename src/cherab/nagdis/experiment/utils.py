"""Utilities for the experiment module."""

import numpy as np
import xarray as xr

__all__ = ["create_images"]


def create_images(ds: xr.Dataset, time_name: str = "time_video") -> xr.DataArray:
    """Create images from the dataset.

    Raw dataset contains camera images as 1-D arrays for each port view.
    This function restores 2-D camera images from the 1-D arrays with masks in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to create images.
    time_name : str, optional
        Name of time dimension followed by the camera images, by default "time_video".

    Returns
    -------
    xarray.DataArray
        2-D Images DataArray with (time, y, x) dimensions.
    """
    # Create images
    images = np.full((ds[time_name].size, ds["y"].size, ds["x"].size), np.nan)
    masks: list[str] = [var for var in ds.data_vars if "mask" in var]  # type: ignore

    for mask in masks:
        images[:, ds[mask]] = ds[mask.split("mask-")[1]]

    return xr.DataArray(images, coords={time_name: ds[time_name], "y": ds["y"], "x": ds["x"]})
