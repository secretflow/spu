# Contribution guidelines

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project.

## Repo layout

- Please see [repo layout](REPO_LAYOUT.md).

## Style

### C++ coding style

In general, please use clang-format to format code, and follow clang-tidy tips.

Most of the code style is derived from the
[Google C++ style guidelines](https://google.github.io/styleguide/cppguide.html), except:

- Exceptions are allowed and encouraged where appropriate.
- Header guards should use `#pragma once`.
- Adopt [camelBack](https://llvm.org/docs/Proposals/VariableNames.html#variable-names-coding-standard-options)
    for function names.
- Use [fixed width integer types](https://en.cppreference.com/w/cpp/types/integer) whenever possible.
- Avoid using size_t on interface APIs.

The compiler portion of the project follows [MLIR style](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide).

### Other tips

- Git commit message should be meaningful, we suggest imperative [keywords](https://github.com/joelparkerhenderson/git_commit_message#summary-keywords).
- Developer must write unit-test (line coverage must be greater than 80%), tests should be deterministic.
- Read awesome [Abseil Tips](https://abseil.io/tips/)

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
         secretflow/spu-ci:latest

# attach to build container
docker exec -it spu-gcc11-anolis-dev-$(whoami) bash
```

#### Linux

```sh
Install gcc>=11.2, cmake>=3.18, ninja, nasm>=2.15, python==3.8, bazel==5.4.1, golang

python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
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
# Be aware, brew may install a newer version of bazel, when that happens bazel will give an error message during build.
# Please follow instructions in the error message to install the required version
brew install bazel cmake ninja libomp go

# For Intel mac only
brew install nasm

# Install python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
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

### Bazel build options

- `--define gperf=on` enable gperf
- `--define tracelog=on` enable link trace log.

### Build docs

```sh
# prerequisite
pip install -U -r docs/requirements.txt

cd docs && make html  # html docs will be in docs/_build/html
```

## Release cycle

SPU recommends users "live-at-head" like [abseil-cpp](https://github.com/abseil/abseil-cpp),
just like abseil, spu also provide Long Term Support Releases to which we backport fixes for severe bugs.

We use the release date as the version number, see [change log](CHANGELOG.md) for example.

## Change log

Please keep updating changes to the staging area of [change log](CHANGELOG.md)
Changelog should contain:

- all public API changes, including new features, breaking changes and deprecations.
- notable internal changes, like performance improvement.
