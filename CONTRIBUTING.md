# Contributing

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

Most of the code style is derived from the [Google C++ style guidelines](https://google.github.io/styleguide/cppguide.html), except:

* Exceptions are allowed and encouraged where appropriate.
* Header guards should use `#pragma once`.
* Adopts [camelBack](https://llvm.org/docs/Proposals/VariableNames.html#variable-names-coding-standard-options) for function names.

The compiler portion of the project follows [MLIR style](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide).

### Other tips

* Git commit message should be meaningful, we suggest imperative [keywords](https://github.com/joelparkerhenderson/git_commit_message#summary-keywords).
* Developer must write unit-test (line coverage must be greater than 80%), tests should be deterministic.
* Read awesome [Abseil Tips](https://abseil.io/tips/)

## Release cycle

SPU recommends users "live-at-head" like [abseil-cpp](https://github.com/abseil/abseil-cpp), just like abseil, spu also provide Long Term Support Releases to which we backport fixes for severe bugs.

We use the release date as the version number, see [change log](CHANGELOG.md) for example.

## Change log

Please keep updating changes to the staging area of [change log](CHANGELOG.md)
Changelog should contain:
- all public API changes, including new features, breaking changes and deprecations.
- notable internal changes, like performance improvement.
