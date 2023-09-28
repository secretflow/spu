# Installation Guidelines

## Environment

### Linux

SPU has been tested with the following settings:

- Anolis OS 8.4 or later
- python3.8
- 8c16g

### MacOS

We have conducted some successful preliminary testings on
macOS Monterey 12.4 with Intel processors and Apple Silicon.

### Docker Image

Please check [official Docker image](https://registry.hub.docker.com/r/secretflow/spu-ci).

## Binaries

### From PyPI

You could install SPU via the [official PyPI package](https://pypi.org/project/spu/).

```bash
pip install spu
```

### From Source

- Install dependencies listed [here](CONTRIBUTING.md#prerequisite).
- For CentOS 7 or Ubuntu 22.04, use corresponding dockerfile below as a reference:
  - [Ubuntu 22.04 reference](https://github.com/secretflow/devtools/blob/main/dockerfiles/ubuntu-base-ci.DockerFile)
  - [CentOS 7 reference](https://github.com/secretflow/devtools/blob/main/dockerfiles/release-ci.DockerFile)
- At the root of repo, run

```bash
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
```
