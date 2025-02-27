"""Module to offer functionalities to perform conditional average of dataset."""

from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

from .utils import create_images

__all__ = ["ConditionalAverage"]


class ConditionalAverage:
    """Class to perform conditional average of dataset.

    Parameters
    ----------
    path : Path | str
        Path to the dataset.
    dt : float
        Time range to perform the conditional average (:math:`\\pm\\Delta t` around the peaked time).
    """

    def __init__(self, path: Path | str, dt: float) -> None:
        self.path = path
        self.dt = dt

        # Load dataset
        self.ds = xr.open_dataset(path)

    def get_peaks_time(
        self,
        signal: str = "I_sat",
        height: ArrayLike | None = 2.5,
        prominence: float | None = None,
    ) -> np.ndarray:
        """Get time of peaks in the dataset.

        Parameters
        ----------
        signal : str
            Signal to find peaks.
        height : array_like, optional
            Factor of standard deviation to set the height of peaks, by default 2.5.
            If a float, the height is set to `mean + height * std`.
            If an array_like with two elements, the height is set to
            `(mean + height[0] * std, mean + height[1] * std)`.
            Otherwise, raise ValueError.
        prominence : float, optional
            Required prominence of peaks, by default the standard deviation of the signal.

        Returns
        -------
        numpy.ndarray
            Time of peaks.
        """
        # Get time of peaks
        end_time = self.ds.time[-1].values
        signal_trimmed = self.ds[signal].sel(time=slice(self.dt, end_time - self.dt))
        mean = signal_trimmed.mean().item()
        std = signal_trimmed.std().item()

        if height is None:
            height = mean + 2.5 * std
        else:
            height = np.asarray_chkfinite(height)
            if height.size == 1:
                height = mean + height * std
            elif height.size == 2:
                height = (mean + height[0] * std, mean + height[1] * std)
            else:
                raise ValueError("height must be a float, tuple of two floats, or None.")

        prominence = std if prominence is None else prominence

        peak_indices, _ = find_peaks(
            signal_trimmed.values,
            height=height,
            prominence=prominence,
        )

        return signal_trimmed.time[peak_indices].values

    def average_per_tau(self, peak_times, d_tau: float, tau_eps: float | None = None) -> xr.Dataset:
        """Average the dataset per tau.

        Average the data :math:`f(t)` like:

        .. math::

            f_\\mathrm{avg}(\\tau) = \\frac{1}{\\#T^{(\\tau)}}\\sum_{t \\in T^{(\\tau)}} f(t + \\tau),

            T^{(\\tau)} \\equiv \\{t' \\equiv t_\\mathrm{peak} + \\tau \\mid t_\\mathrm{peak}\\in T_\\mathrm{peak} \\land t' \\in T_\\mathrm{video}\\},

        where :math:`\\tau \\in \\{-\\Delta t, -\\Delta t + \\Delta \\tau, \\ldots, \\Delta t\\}`,
        :math:`T_\\mathrm{peak}` is the set of peak times from the waveform signal, and
        :math:`T_\\mathrm{video}` is the set of video times.
        This function enables to enhance the :math:`\\tau` resolution even if the frequency of the
        video dataset is lower than the signal dataset.

        Parameters
        ----------
        peak_times : array_like
            Set of peak times :math:`T_\\mathrm{peak}`.
        d_tau : float
            Time interval of tau :math:`\\Delta \\tau`.
        tau_eps : float, optional
            Tolerance of video dataset time to match with signal dataset time, by default `d_tau`.

        Return
        ------
        xarray.Dataset
            Averaged dataset.
        """
        tau_eps = d_tau if tau_eps is None else tau_eps
        taus = np.linspace(-self.dt, self.dt, round(2 * self.dt / d_tau) + 1, endpoint=True)

        # Create empty dataset
        ds_avg = (
            xr.zeros_like(self.ds)
            .isel(time=slice(0, taus.size))
            .isel(time_video=slice(0, taus.size))
            .assign_coords(time=taus)
            .assign_coords(time_video=taus)
            .rename(time="tau")
            .rename(time_video="tau_video")
        )
        ds_avg.coords["tau"].attrs["units"] = "s"
        ds_avg.coords["tau_video"].attrs["units"] = "s"

        # Drop variables that does not have tau or tau_video as dimension
        ds_avg = ds_avg.drop_vars(
            [
                var
                for var in ds_avg.data_vars
                if "tau" not in ds_avg[var].dims and "tau_video" not in ds_avg[var].dims
            ]
        )

        # Drop variables that does not have time and tim_video as dimension
        ds = self.ds.drop_vars(
            [
                var
                for var in self.ds.data_vars
                if "time" not in self.ds[var].dims and "time_video" not in self.ds[var].dims
            ]
        )

        # Separate the dataset into two parts (signal and video)
        ds_signal = ds.drop_vars(
            [var for var in ds.data_vars if "time_video" in ds[var].dims]
        ).drop("time_video")
        ds_video = ds.drop_vars(
            [var for var in ds.data_vars if "time_video" not in ds[var].dims]
        ).drop("time")

        ds_avg_signal = ds_avg.drop_vars([var for var in ds_avg.data_vars if ds_avg[var].ndim == 2])
        ds_avg_video = ds_avg.drop_vars([var for var in ds_avg.data_vars if ds_avg[var].ndim == 1])

        # Average the dataset per tau
        samples = np.zeros_like(taus)

        for i, tau in enumerate(taus):
            times = peak_times + tau

            # Average the signal dataset
            ds_avg_signal.loc[dict(tau=tau)] = ds_signal.reindex(time=times, method=None).mean(
                dim="time"
            )

            # Average the video dataset and count the number of non-NaN values
            _d = ds_video.reindex(time_video=times, method="nearest", tolerance=tau_eps)
            ds_avg_video.loc[dict(tau_video=tau)] = _d.mean(dim="time_video")
            samples[i] = _d.count(dim="time_video")["port-1"][0].item()

        # Update the dataset
        ds_avg.update(ds_avg_signal)
        ds_avg.update(ds_avg_video)

        # Merge variables attributes
        for var in ds_avg.data_vars:
            ds_avg[var].attrs.update(ds[var].attrs)

        # Add mask variables
        masks = [var for var in self.ds.data_vars if "mask" in var]
        for mask in masks:
            ds_avg[mask] = self.ds[mask]

        # Add wireframe
        if "wireframe" in self.ds:
            ds_avg["wireframe"] = self.ds["wireframe"]

        # Add number of peaks
        ds_avg = ds_avg.assign_attrs(
            num_peaks=peak_times.size,
            description="Number of peaks",
        )

        # Add samples to the dataset
        ds_avg = ds_avg.assign(
            samples=("tau", samples, {"description": "Number of samples per tau"})
        )

        # Add number of peaks
        ds_avg = ds_avg.assign_attrs(
            num_peaks=len(peak_times),
            description="Number of peaks",
        )

        # rename tau_video to tau
        ds_avg = ds_avg.drop_vars("tau_video").rename(tau_video="tau")

        return ds_avg

    def average(self, peak_time, time_eps: float | None = None) -> xr.Dataset:
        """Average the dataset.

        Average the data :math:`f(t)` like:

        .. math::

            f_\\mathrm{avg}(\\tau) = \\frac{1}{\\#T}\\sum_{t \\in T} f(t + \\tau),

            T \\equiv \\{t' \\mid t'\\in T_\\mathrm{peak}\\cap T_\\mathrm{video}\\},

        where :math:`\\tau \\in [-\\Delta t, \\Delta t]`, :math:`T_\\mathrm{peak}` is the set of
        peak times from the waveform signal, and :math:`T_\\mathrm{video}` is the set of video times.

        Parameters
        ----------
        peak_time : array_like
            Set of peak times :math:`T_\\mathrm{peak}`.
        time_eps : float, optional
            Tolerance of peak time corresponding video time, by default `0.5 * d_t`.

        Returns
        -------
        xarray.Dataset
            Averaged dataset.
        """
        d_t = self.ds.time.values[1] - self.ds.time.values[0]
        d_t_video = self.ds.time_video.values[1] - self.ds.time_video.values[0]

        time_eps = d_t * 0.5 if time_eps is None else time_eps

        taus = np.linspace(-self.dt, self.dt, round(2 * self.dt / d_t) + 1, endpoint=True)
        taus_video = np.linspace(
            -self.dt, self.dt, round(2 * self.dt / d_t_video) + 1, endpoint=True
        )

        # Extract peak times that match video times
        peak_times = (
            self.ds["time_video"]
            .reindex(time_video=peak_time, method="nearest", tolerance=time_eps)
            .dropna("time_video")
            .values
        )

        # Create empty dataset
        ds_avg = (
            xr.zeros_like(self.ds)
            .isel(time=slice(0, taus.size))
            .isel(time_video=slice(0, taus_video.size))
            .assign_coords(time=taus)
            .assign_coords(time_video=taus_video)
            .rename(time="tau")
            .rename(time_video="tau_video")
        )
        ds_avg.coords["tau"].attrs["units"] = "s"
        ds_avg.coords["tau_video"].attrs["units"] = "s"

        # Drop variables that does not have tau or tau_video as dimension
        ds_avg = ds_avg.drop_vars(
            [
                var
                for var in ds_avg.data_vars
                if "tau" not in ds_avg[var].dims and "tau_video" not in ds_avg[var].dims
            ]
        )
        ds = self.ds.drop_vars(
            [
                var
                for var in self.ds.data_vars
                if "time" not in self.ds[var].dims and "time_video" not in self.ds[var].dims
            ]
        )

        # Separate the dataset into two parts (signal and video)
        ds_signal = ds.drop_vars(
            [var for var in ds.data_vars if "time_video" in ds[var].dims]
        ).drop("time_video")
        ds_video = ds.drop_vars(
            [var for var in ds.data_vars if "time_video" not in ds[var].dims]
        ).drop("time")

        ds_avg_signal = ds_avg.drop_vars([var for var in ds_avg.data_vars if ds_avg[var].ndim == 2])
        ds_avg_video = ds_avg.drop_vars([var for var in ds_avg.data_vars if ds_avg[var].ndim == 1])

        # Average the dataset per peak time
        eps = d_t * 0.5
        for t in peak_times:
            # Sum the signal dataset between t - dt and t + dt
            ds_avg_signal += (
                ds_signal.sel(time=slice(t - (self.dt + eps), t + self.dt + eps))
                .assign_coords(time=taus)
                .rename(time="tau")
            )

            # Sum the video dataset between t - dt and t + dt
            ds_avg_video += (
                ds_video.sel(time_video=slice(t - (self.dt + time_eps), t + self.dt + time_eps))
                .assign_coords(time_video=taus_video)
                .rename({"time_video": "tau_video"})
            )

        # Average the dataset
        ds_avg_signal /= len(peak_times)
        ds_avg_video /= len(peak_times)

        # Update the dataset
        ds_avg.update(ds_avg_signal)
        ds_avg.update(ds_avg_video)

        # Merge variables attributes
        for var in ds_avg.data_vars:
            ds_avg[var].attrs.update(ds[var].attrs)

        # Add mask variables
        masks = [var for var in self.ds.data_vars if "mask" in var]
        for mask in masks:
            ds_avg[mask] = self.ds[mask]

        # Add wireframe
        if "wireframe" in self.ds:
            ds_avg["wireframe"] = self.ds["wireframe"]

        # Add number of peaks
        ds_avg = ds_avg.assign_attrs(
            num_peaks=len(peak_times),
            description="Number of peaks",
        )

        return ds_avg

    @staticmethod
    def create_images(ds: xr.Dataset, time_name: str = "tau") -> xr.DataArray:
        """Create images from the dataset.

        This function is a proxy for the `create_images` function in the `utils` module.
        """
        return create_images(ds, time_name)

    @staticmethod
    def create_t_y_contour(ds: xr.Dataset, time_name: str = "tau") -> xr.DataArray:
        """Create t-y contour from the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to create time-axis and y-axis contour.
        time_name : str, optional
            Name of time dimension, by default "tau".

        Returns
        -------
        xarray.DataArray
            A t-y contour DataArray.
        """
        masks: list[str] = [var for var in ds.data_vars if "mask" in var]  # type: ignore

        images = np.full((len(masks), ds["tau"].size, ds["y"].size, ds["x"].size), np.nan)
        for i_port, mask in enumerate(masks):
            images[i_port, :, ds[mask]] = ds[mask.split("mask-")[1]].T

        return xr.DataArray(
            images,
            coords={
                "port": [1, 2, 3, 4, 5],
                time_name: ds[time_name],
                "y": ds["y"],
                "x": ds["x"],
            },
        ).mean("x", skipna=True)
