# Installation Guidelines

## Environment

### Linux

SPU has been tested with the following settings:

- Ubuntu 22.04
- python3.8
- 8c16g

### MacOS

We have conducted some successful preliminary testings on
macOS Monterey 14.1 with Apple Silicon.

### Docker Image

Please check [official Docker image](https://hub.docker.com/r/secretflow/ubuntu-base-ci).

## Binaries

### From PyPI

You could install SPU via the [official PyPI package](https://pypi.org/project/spu/).

```bash
pip install spu
```

### From Source

- Install dependencies listed [here](https://github.com/secretflow/spu/blob/main/CONTRIBUTING.md#prerequisite).
- For CentOS 7 or Ubuntu 22.04, use corresponding dockerfile below as a reference:
  - [Ubuntu 22.04 reference](https://github.com/secretflow/devtools/blob/main/dockerfiles/ubuntu-base-ci.DockerFile)
  - [CentOS 7 reference](https://github.com/secretflow/devtools/blob/main/dockerfiles/release-ci.DockerFile)
- At the root of repo, run

```bash
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
```

- Once GCC/bazel/python/Xcode version or other environment settings have changed, please run the following command to ensure a clean build

```bash
bazel clean --expunge
```

#### Build with GPU support

```bash
export ENABLE_GPU_BUILD=1 && python setup.py bdist_wheel
```
