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

Please check [official Docker image](https://registry.hub.docker.com/r/secretflow/secretflow-gcc11-anolis-dev).

## Binaries

### From PyPI

You could install SPU via the [official PyPI package](https://pypi.org/project/spu/)

```bash
pip install spu
```

### From Source

At the root of repo, run

```bash
sh build_and_packaging.sh -i
```
