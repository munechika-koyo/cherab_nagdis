"""Module to fetch files from the remote server using the SFTP downloader."""

from __future__ import annotations

import os

import pooch
from pooch import SFTPDownloader
from rich.console import Console
from rich.table import Table

__all__ = ["fetch_file", "show_registries"]


HOSTNAME = os.environ.get("SSH_RAYTRACE_HOSTNAME", default="sftp://example.com/")
USERNAME = os.environ.get("SSH_RAYTRACE_USERNAME", default="username")
PASSWORD = os.environ.get("SSH_RAYTRACE_PASSWORD", default="password")

# Registry of the datasets
REGISTRIES = {
    # calcam calibration data
    "20240705_mod.ccc": "70b6493c4a2f8f4cdc873f378fd32f9b0db5748293612238775ab01f40055af7",
    # calcam cad data
    "machine/nagdis-ii.ccm": "255b9b922ec17cedca70436d1639e6a93f08b5df6b5ee2ca8457182514fde873",
    # raysect machine mesh data
    "machine/gate_valve.rsm": "a5ae7db5b594fbf889d27f6c9b2d70cb7dd32869203222bf942121d30960bc0a",
    "machine/vessel_lower_fine.rsm": "621dd181dff6e550921a3cfc19f6a1cd203f8c3cc2da7b8d28c22ff58d8f23d0",
    "machine/vessel_upper_fine.rsm": "fe270ac366bead57f0b520d61e524e4dfca4578b5a6512c5bd6e7dd6e1e2d56d",
    "machine/vessel_lower.rsm": "70d96acd19e212845404765ecb10dd34a3162ac84d3cd58f59170d00ba0c9a24",
    "machine/vessel_upper.rsm": "39cf6dcfbcd7718c8ed1c0067e6930f06db22b98cf6b95aa6486cba41dc34688",
    "machine/coils.rsm": "16318505135dd4f22577a097df7560f95de12fa9a3dce6baf89dcd2dabb157cf",
    "machine/coils_v2.rsm": "953ccbce46e1174a8d3ef37684baee6d86dd19c72db32d00186fb07d4760733e",
    # material data
    "materials/sus316L.json": "0f473be0d3f9efc88d11d3cbacbc5a8596ff2b39dcb2dbd561d0fa116de5f301",
}


def show_registries() -> None:
    """Show the registries of the datasets."""
    table = Table(title="Registries", show_lines=True)
    table.add_column("File Name", style="cyan", justify="left")
    table.add_column("SHA256", style="dim", justify="left")

    for name, sha256 in REGISTRIES.items():
        table.add_row(name, sha256)
    console = Console()
    console.print(table)


def fetch_file(
    name: str,
    host: str = HOSTNAME,
    username: str = USERNAME,
    password: str = PASSWORD,
) -> str:
    """Fetch the file from the remote server using the configured SFTP downloader.

    Fetched data will be stored in the cache directory like `~/.cache/cherab/nagdis`.

    Parameters
    ----------
    name : str
        Name of the file to fetch.
    host : str, optional
        Host name of the server, by default ``sftp://example.com/``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_HOSTNAME`.
        Host name should be in the format ``sftp://{host's name or ip}/{directories}``.
    username : str, optional
        Username to authenticate with the server, by default ``username``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_USERNAME`.
    password : str, optional
        Password to authenticate with the server, by default ``password``.
        This value is adaptable from the environment variable `SSH_RAYTRACE_PASSWORD`.

    Returns
    -------
    str
        Path to the fetched file.
    """
    pup = pooch.create(
        path=pooch.os_cache("cherab/nagdis"),
        base_url=host,
        registry=REGISTRIES,
    )

    downloader = SFTPDownloader(
        username=username,
        password=password,
        progressbar=True,
        timeout=5,
    )
    return pup.fetch(name, downloader=downloader)
