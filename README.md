# CHERAB-NAGDIS

| | |
| ------- | ------- |
| CI/CD   | [![CI - Test][test-badge]][test] [![pre-commit.ci status][pre-commit-ci-badge]][pre-commit-ci]
| Meta    | [![DOI][DOI-badge]][DOI] [![License - MIT][License-badge]][License] [![Pixi Badge][pixi-badge]][pixi-url] |

<!-- CI/CD -->
[test-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_nagdis/test.yml?style=flat-square&label=test&logo=github
[test]: https://github.com/munechika-koyo/cherab_nagdis/actions/workflows/test.yml
[pre-commit-ci-badge]: https://results.pre-commit.ci/badge/github/munechika-koyo/cherab_nagdis/main.svg
[pre-commit-ci]: https://results.pre-commit.ci/latest/github/munechika-koyo/cherab_nagdis/main
<!-- Meta -->
[DOI-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.10118752.svg
[DOI]: https://doi.org/10.5281/zenodo.10118752
[License-badge]: https://img.shields.io/github/license/munechika-koyo/cherab_nagdis?style=flat-square
[License]: https://opensource.org/licenses/MIT
[pixi-badge]:https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh

----

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
