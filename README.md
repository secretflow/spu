# SPU: Secure Processing Unit

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/secretflow/spu/tree/beta.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/secretflow/spu/tree/beta)

SPU (Secure Processing Unit) aims to be a `provable`, `measurable` secure computation device, which provides computation ability while keeping your private data protected.

## Project status

Currently, we mainly focus on `provable` security. It contains a secure runtime that evaluates [XLA](https://www.tensorflow.org/xla/operation_semantics)-like tensor operations, which use [MPC](https://en.wikipedia.org/wiki/Secure_multi-party_computation) as the underline evaluation engine to protect privacy information.

## Contents
- [Documentation](https://secretflow.readthedocs.io)
- [Roadmap](TBD)
- [Build and test](#Build)
- [FAQ](#FAQ)

## Build

### Prerequisite

#### Docker
```sh
## start container
docker run -d -it --name spu-gcc11-anolis-dev-$(whoami) \
         --mount type=bind,source="$(pwd)",target=/home/admin/dev/ \
         -w /home/admin/dev \
         --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
         --cap-add=NET_ADMIN \
         --privileged=true \
         registry.hub.docker.com/secretflow/spu-gcc11-anolis-dev:latest

# attach to build container
docker exec -it spu-gcc11-anolis-dev-$(whoami) bash
```

#### Linux

```sh
Install gcc>=11.2, cmake>=3.18, ninja, nasm>=2.15, python==3.8, bazel==5.1.1

python3 -m pip install -r docker/requirements.txt
```

#### macOS

```sh
# macOS >= 11, Xcode >=13.0

# Install Xcode
https://apps.apple.com/us/app/xcode/id497799835?mt=12

# Select Xcode toolchain version
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# Install homebrew
https://brew.sh/

# Install dependencies
brew install bazel cmake ninja nasm

# Install python dependencies
python3 -m pip install -r docker/requirements.txt
```

### Build & UnitTest

``` sh

# build as debug
bazel build //... -c dbg

# build as release
bazel build //... -c opt

# test
bazel test //...

# [optional] build & test with ASAN or UBSAN, for macOS users please use configs with macOS prefix
bazel test //... --config=[macos-]asan
bazel test //... --config=[macos-]ubsan
```


### Build docs

```sh
# prerequisite
pip install -U sphinx

cd docs & make html  # html docs will be in docs/_build/html
```

## FAQ

> How can I use SPU?

SPU could be treated as a programmable device, it's not designed to be used directly. Normally we use SecretFlow framework, which use SPU as the underline secure computing device.

SPU python package also contains a simple distributed module to demo SPU usage, but it's **NOT designed for production** due to system security and performance concerns, please **DO NOT** use it directly in production.

## Acknowledgement

We thank the significant contributions made by [Alibaba Gemini Lab](https://alibaba-gemini-lab.github.io).
