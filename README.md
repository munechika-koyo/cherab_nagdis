# CHERAB-NAGDIS

|         |                                                                                                                       |
| ------- | --------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CI][CI-badge]][CI] [![PyPI Publish][PyPI-publish-badge]][PyPi-publish] [![codecov][codecov-badge]][codecov]        |
| Docs    | [![Read the Docs (version)][Docs-dev-badge]][Docs-dev] [![Read the Docs (version)][Docs-release-badge]][Docs-release] |
| Package | [![PyPI - Version][PyPI-badge]][PyPI] [![Conda][Conda-badge]][Conda] [![PyPI - Python Version][Python-badge]][PyPI]   |
| Meta    | [![DOI][DOI-badge]][DOI] [![License - MIT][License-badge]][License] [![Pixi Badge][pixi-badge]][pixi-url]             |

[CI-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_nagdis/ci.yaml?style=flat-square&logo=GitHub&label=CI
[CI]: https://github.com/munechika-koyo/cherab_nagdis/actions/workflows/ci.yaml
[PyPI-publish-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_nagdis/pypi-publish.yaml?style=flat-square&label=PyPI%20Publish&logo=github
[PyPI-publish]: https://github.com/munechika-koyo/cherab_nagdis/actions/workflows/pypi-publish.yaml
[codecov-badge]: https://img.shields.io/codecov/c/github/munechika-koyo/cherab_nagdis?token=05LZGWUUXA&style=flat-square&logo=codecov
[codecov]: https://codecov.io/github/munechika-koyo/cherab_nagdis
[Docs-dev-badge]: https://img.shields.io/readthedocs/cherab-nagdis/latest?style=flat-square&logo=readthedocs&label=dev%20docs
[Docs-dev]: https://cherab-nagdis.readthedocs.io/en/latest/?badge=latest
[Docs-release-badge]: https://img.shields.io/readthedocs/cherab-nagdis/stable?style=flat-square&logo=readthedocs&label=release%20docs
[Docs-release]: https://cherab-nagdis.readthedocs.io/en/stable/?badge=stable
[PyPI-badge]: https://img.shields.io/pypi/v/cherab-nagdis?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[PyPI]: https://pypi.org/project/cherab-nagdis/
[Conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-nagdis?logo=conda-forge&style=flat-square
[Conda]: https://prefix.dev/channels/conda-forge/packages/cherab-nagdis
[Python-badge]: https://img.shields.io/pypi/pyversions/cherab-nagdis?logo=Python&logoColor=gold&style=flat-square
[DOI-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.14929182.svg
[DOI]: https://doi.org/10.5281/zenodo.14929182
[License-badge]: https://img.shields.io/github/license/munechika-koyo/cherab_nagdis?style=flat-square
[License]: https://opensource.org/licenses/MIT
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh

---

This repository contains the NAGDIS-II machine-dependent extensions of [`cherab`](https://www.cherab.info/) code.

## Table of Contents

- [Get Started](#installation)
- [License](#license)

## Get Started

### Task-based execution

We offer some tasks to execute programs in CLI.
You can see the list of tasks using [pixi](https://pixi.sh) command.

```console
pixi task list
```

If you want to execute a task, you can use the following command.

```console
pixi run <task_name>
```

### Notebooks

We provide some notebooks to demonstrate the usage of the CHERAB-NAGDIS code.
To launch the Jupyter lab server, you can use the following command.

```console
pixi run lab
```

Then, you can access the Jupyter lab server from your web browser.

## License

`cherab-nagdis` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
