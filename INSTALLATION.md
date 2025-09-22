# Installation Guidelines

There are three ways to install SPU: using official Docker image, installing from PyPI, and building from source.

## Environment

### Linux

SPU has been tested with the following settings:

- Ubuntu 22.04
- Python 3.10 and 3.11

### macOS

We have conducted some successful preliminary testings on
macOS Monterey 14.1 with Apple Silicon.

## Docker Image

Please check [official Docker image](https://hub.docker.com/r/secretflow/ubuntu-base-ci).

## Installing From PyPI

You could install SPU via the [official PyPI package](https://pypi.org/project/spu/).
Note that SPU current only support Python 3.10 and 3.11.

```bash
pip install spu
```






### From Source

It is recommended to install in a virtual environment, e.g.,

```bash
conda create -n my-env python=3.11
conda activate my-env
pip install spu
```

## Building From Source

- Install dependencies listed [here](https://github.com/secretflow/spu/blob/main/CONTRIBUTING.md#prerequisite).
- For CentOS 7 or Ubuntu 22.04, use corresponding dockerfile below as a reference:
  - [Ubuntu 22.04 reference](https://github.com/secretflow/devtools/blob/main/dockerfiles/ubuntu-base-ci.DockerFile)
  - [CentOS 7 reference](https://github.com/secretflow/devtools/blob/main/dockerfiles/release-ci.DockerFile)
- At the root of repo, run

```bash
conda create -n build python=3.11 -y
conda activate build

python3 -m pip install build
python3 -m build --wheel
ls dist/spu*.whl

pip install dist/spu*.whl --force-reinstall
```

- Note that:
  - This will build with Python 3.11 by default. See [below](#build-with-specified-python-version) for specifing Python version when building.
  - The Python version used for building (specified in `bazelisk`) must match the Python version used for `pip install` (can be checked using `pip -V`).
  It is recommended to install in a virtual environment.
- Once GCC/bazel/python/Xcode version or other environment settings have changed, please run the following command to ensure a clean build

```bash
bazelisk clean --expunge
```

### Build with GPU support

This requires CUDA Toolkit to be installed.

```bash
bazelisk build //:spu_wheel -c opt --config=gpu
```

### Build with specified python version

```bash
# build with python 3.10
bazelisk build //:spu_wheel -c opt --@rules_python//python/config_settings:python_version=3.10

# build with python 3.11
bazelisk build //:spu_wheel -c opt --@rules_python//python/config_settings:python_version=3.11
```
