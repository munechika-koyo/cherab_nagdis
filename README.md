# CHERAB-NAGDIS

<!-- BEGIN-HEADER -->

|         |                                                                                                                       |
| ------- | --------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CI][ci-badge]][ci] [![PyPI Publish][PyPI-publish-badge]][PyPi-publish] [![codecov][codecov-badge]][codecov]        |
| Docs    | [![Read the Docs (version)][Docs-dev-badge]][Docs-dev] [![Read the Docs (version)][Docs-release-badge]][Docs-release] |
| Package | [![PyPI - Version][pypi-badge]][pypi] [![Conda][conda-badge]][conda] [![PyPI - Python Version][python-badge]][pypi]   |
| Meta    | [![DOI][DOI-badge]][DOI] [![License - MIT][license-badge]][license] [![Pixi Badge][pixi-badge]][pixi-url]             |

[ci]: https://github.com/munechika-koyo/cherab_nagdis/actions/workflows/ci.yaml
[ci-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_nagdis/ci.yaml?style=flat-square&logo=GitHub&label=CI
[codecov]: https://codecov.io/github/munechika-koyo/cherab_nagdis
[codecov-badge]: https://img.shields.io/codecov/c/github/munechika-koyo/cherab_nagdis?token=05LZGWUUXA&style=flat-square&logo=codecov
[conda]: https://prefix.dev/channels/conda-forge/packages/cherab-nagdis
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-nagdis?logo=conda-forge&style=flat-square
[Docs-dev-badge]: https://img.shields.io/readthedocs/cherab-nagdis/latest?style=flat-square&logo=readthedocs&label=dev%20docs
[Docs-dev]: https://cherab-nagdis.readthedocs.io/en/latest/?badge=latest
[Docs-release-badge]: https://img.shields.io/readthedocs/cherab-nagdis/stable?style=flat-square&logo=readthedocs&label=release%20docs
[Docs-release]: https://cherab-nagdis.readthedocs.io/en/stable/?badge=stable
[PyPI-badge]: https://img.shields.io/pypi/v/cherab-nagdis?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[docs-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_nagdis/docs.yml?style=flat-square&logo=GitHub&label=Docs
[DOI-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.18257046.svg
[DOI]: https://doi.org/10.5281/zenodo.18257046
[License-badge]: https://img.shields.io/github/license/munechika-koyo/cherab_nagdis?style=flat-square
[License]: https://opensource.org/licenses/MIT
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh
[pypi]: https://pypi.org/project/cherab-nagdis/
[pypi-badge]: https://img.shields.io/pypi/v/cherab-nagdis?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[pypi-publish]: https://github.com/munechika-koyo/cherab_nagdis/actions/workflows/pypi-publish.yaml
[pypi-publish-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_nagdis/pypi-publish.yaml?event=release&style=flat-square&logo=github&label=PyPI%20Publish
[python-badge]: https://img.shields.io/pypi/pyversions/cherab-nagdis?logo=Python&logoColor=gold&style=flat-square

---

This repository contains the NAGDIS-II machine-dependent extensions of [`cherab`](https://www.cherab.info/) code.

<!-- END-HEADER -->

## Get Started

### Task-based execution

We offer some tasks to execute programs in CLI.
You can see the list of tasks using [pixi](https://pixi.sh) command.

```bash
pixi task list
```

If you want to execute a task, you can use the following command.

```bash
pixi run <task_name>
```

If you want to run a python script stored in `scripts/` directory, you can use the following command.

```bash
pixi run python script/<script_name>.py
```

### Notebooks

We provide some notebooks to demonstrate the usage of the CHERAB-NAGDIS code.
To launch the Jupyter lab server, you can use the following command.

```bash
pixi run lab
```

Then, you can access the Jupyter lab server from your web browser.

## 🌐 Installation

You can install the package from PyPI:

```bash
pip install cherab-nagdis
```

Or from Conda:

```bash
mamba install -c conda-forge cherab-nagdis
```

## 📝 Documentation

The [documentation](https://cherab-lhd.readthedocs.io/) is made with [Sphinx](https://www.sphinx-doc.org/en/master/) and hosted on [Read the Docs](https://readthedocs.org/).
There are two versions of the documentation:

- [Development](https://cherab-lhd.readthedocs.io/en/latest/)
- [Release](https://cherab-lhd.readthedocs.io/en/stable/)

## 📄 License

`cherab-nagdis` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
