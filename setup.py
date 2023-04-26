# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ideas borrowed from: https://github.com/ray-project/ray/blob/master/python/setup.py

import io
import logging
import os
import re
import shutil
import subprocess
import sys
import setuptools
import setuptools.command.build_ext
import platform

logger = logging.getLogger(__name__)

# 3.8 is the minimum python version we can support
SUPPORTED_PYTHONS = [(3, 8), (3, 9), (3, 10), (3, 11)]

BAZEL_MAX_JOBS = os.getenv("BAZEL_MAX_JOBS")
ROOT_DIR = os.path.dirname(__file__)
SKIP_BAZEL_CLEAN = os.getenv("SKIP_BAZEL_CLEAN")


pyd_suffix = ".so"


def find_version(*filepath):
    # Extract version information from filepath
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def read_requirements(*filepath):
    requirements = []
    with open(os.path.join(ROOT_DIR, *filepath)) as file:
        requirements = file.read().splitlines()
    return requirements


class SetupSpec:
    def __init__(self, name: str, description: str):
        self.name: str = name
        self.version = find_version("spu", "version.py")
        self.description: str = description
        self.files_to_include: list = []
        self.install_requires: list = []
        self.extras: dict = {}

    def get_packages(self):
        return setuptools.find_packages()


setup_spec = SetupSpec(
    "spu",
    "SPU aims to be a 'provable', 'measurable' secure computation device.",
)

# Ideally, we could include these files by putting them in a
# MANIFEST.in or using the package_data argument to setup, but the
# MANIFEST.in gets applied at the very beginning when setup.py runs
# before these files have been created, so we have to move the files
# manually.

# NOTE: The lists below must be kept in sync with spu/BUILD.bazel.
spu_lib_files = [
    "bazel-bin/spu/libspu" + pyd_suffix,
]

# These are the directories where automatically generated Python protobuf
# bindings are created.
generated_python_directories = [
    "bazel-bin/spu",
    "bazel-bin/libspu",
    "bazel-bin/libspu/pir",
    "bazel-bin/libspu/psi",
    "bazel-bin/spu/utils",
]

setup_spec.install_requires = read_requirements('requirements.txt')

files_to_remove = []


# Calls Bazel in PATH
def bazel_invoke(invoker, cmdline, *args, **kwargs):
    try:
        result = invoker(['bazel'] + cmdline, *args, **kwargs)
        return result
    except IOError:
        raise


def build(build_python, build_cpp):
    if tuple(sys.version_info[:2]) not in SUPPORTED_PYTHONS:
        msg = (
            "Detected Python version {}, which is not supported. "
            "Only Python {} are supported."
        ).format(
            ".".join(map(str, sys.version_info[:2])),
            ", ".join(".".join(map(str, v)) for v in SUPPORTED_PYTHONS),
        )
        raise RuntimeError(msg)

    bazel_env = dict(os.environ, PYTHON3_BIN_PATH=sys.executable)

    bazel_flags = ["--verbose_failures"]
    if BAZEL_MAX_JOBS:
        n = int(BAZEL_MAX_JOBS)  # the value must be an int
        bazel_flags.append(f"--jobs={n}")

    bazel_precmd_flags = []

    bazel_targets = []
    bazel_targets += ["//spu:init", "//spu/utils:distributed"] if build_python else []
    bazel_targets += ["//spu:api"] if build_cpp else []

    bazel_flags.extend(["-c", "opt"])

    if sys.platform == "linux":
        bazel_flags.extend(["--config=linux-release"])

    return bazel_invoke(
        subprocess.check_call,
        bazel_precmd_flags + ["build"] + bazel_flags + ["--"] + bazel_targets,
        env=bazel_env,
    )


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix) :]


def copy_file(target_dir, filename, rootdir):
    source = os.path.relpath(filename, rootdir)
    destination = os.path.join(target_dir, remove_prefix(source, 'bazel-bin/'))

    # Create the target directory if it doesn't already exist.
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        print(f"Copy file from {source} to {destination}")
        shutil.copy(source, destination, follow_symlinks=True)
        return 1
    return 0


def remove_file(target_dir, filename):
    file = os.path.join(target_dir, filename)
    if os.path.exists(file):
        print(f"delete {file}")
        os.remove(file)
        return 1
    return 0


def pip_run(build_ext):
    build(True, True)

    setup_spec.files_to_include += spu_lib_files

    # Copy over the autogenerated protobuf Python bindings.
    for directory in generated_python_directories:
        for filename in os.listdir(directory):
            if filename[-3:] == ".py":
                setup_spec.files_to_include.append(os.path.join(directory, filename))

    copied_files = 0
    for filename in setup_spec.files_to_include:
        copied_files += copy_file(build_ext.build_lib, filename, ROOT_DIR)
    print("# of files copied to {}: {}".format(build_ext.build_lib, copied_files))

    deleted_files = 0
    for filename in files_to_remove:
        deleted_files += remove_file(build_ext.build_lib, filename)
    print("# of files deleted in {}: {}".format(build_ext.build_lib, deleted_files))


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        return pip_run(self)


class BinaryDistribution(setuptools.Distribution):
    def has_ext_modules(self):
        return True


# Ensure no remaining lib files.
build_dir = os.path.join(ROOT_DIR, "build")
if os.path.isdir(build_dir):
    shutil.rmtree(build_dir)

if not SKIP_BAZEL_CLEAN:
    bazel_invoke(subprocess.check_call, ['clean'])

# Default Linux platform tag
plat_name = "manylinux2014_x86_64"

if sys.platform == "darwin":
    # Due to a bug in conda x64 python, platform tag has to be 10_16 for X64 wheel
    if platform.machine() == "x86_64":
        plat_name = "macosx_10_16_x86_64"
    else:
        plat_name = "macosx_11_0_arm64"

setuptools.setup(
    name=setup_spec.name,
    version=setup_spec.version,
    author="SecretFlow Team",
    author_email='secretflow-contact@service.alipay.com',
    description=(setup_spec.description),
    long_description=io.open(
        os.path.join(ROOT_DIR, "README.md"), "r", encoding="utf-8"
    ).read(),
    long_description_content_type='text/markdown',
    url="https://github.com/secretflow/spu",
    keywords=("spu mpc secretflow compiler vm ABY3 secure computation"),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=setup_spec.get_packages(),
    cmdclass={"build_ext": build_ext},
    # The BinaryDistribution argument triggers build_ext.
    distclass=BinaryDistribution,
    install_requires=setup_spec.install_requires,
    setup_requires=["wheel"],
    extras_require=setup_spec.extras,
    license="Apache 2.0",
    options={'bdist_wheel': {'plat_name': plat_name}},
)
