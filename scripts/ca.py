#!/usr/bin/env -S pixi run python
"""Example script for conditional averaging of dataset.

This script demonstrates how to use the `.ConditionalAverage` class to find peaks in the dataset and
average them.
"""

from argparse import ArgumentParser
from pathlib import Path

from cherab.nagdis.experiment.conditional_average import ConditionalAverage

ROOT = Path(__file__).parents[1]


# Parse command line arguments
parser = ArgumentParser(description="Conditional averaging of dataset")
parser.add_argument("file_path", type=str, help="Path to the dataset file from root of the project")
args = parser.parse_args()

# Use the provided file path
file_path = ROOT / Path(args.file_path)

# Load dataset
ca = ConditionalAverage(file_path, 300)
print(ca.ds)


# Find peaks
height = (2, 3)
# height = (3, 4)
# height = (3, 5)
peak_time = ca.get_peaks_time(height=height)


# %%
# Simple averaging (match the peak time to camera frame)
# ------------------------------------------------------
# ds_avg = ca.average(peak_time)
# ds_avg.to_netcdf("./data/sccm430/n=3_G32/ca_simple.nc")

# %%
# Complicated averaging (match the tau to camera frame)
# ------------------------------------------------------
ds_avg = ca.average_per_tau(peak_time, d_tau=1)
ds_avg.to_netcdf(file_path.parent / f"ca_s{height[0]}{height[1]}.nc")

# %%
# Moving average
# --------------
# Averaging the dataset with a moving average of 10 points
ds_avg_ma10 = ds_avg.rolling(tau=10, center=True).mean().dropna(dim="tau")
ds_avg_ma10.to_netcdf(file_path.parent / f"ca_s{height[0]}{height[1]}_ma10.nc")
