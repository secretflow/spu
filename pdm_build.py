# Copyright 2021 Ant Group Co., Ltd.
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

import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_version() -> str:
    import re

    version_file = "src/libspu/version.h"
    version_pattern = r'#define SPU_VERSION\s+"([^"]+)"'

    try:
        with open(version_file, "r") as file:
            content = file.read()
            match = re.search(version_pattern, content)
            if match:
                return match.group(1)
            else:
                raise ValueError(f"cannot find version from {version_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"{version_file} not exits.")
    except Exception as e:
        raise Exception(f"parse version fail: {e}")


def _build_lib(libname: str, force: bool = False) -> Path:
    target = f"//spu:{libname}"
    dst_path = Path(f"spu/{libname}.so")

    if dst_path.exists():
        if force:
            dst_path.unlink()
        else:
            print(f"{dst_path} exists, ignore build")
            return dst_path

    ver_info = sys.version_info
    version = f"{ver_info.major}.{ver_info.minor}"
    args = [
        "bazelisk",
        "build",
        target,
        f"--@rules_python//python/config_settings:python_version={version}",
    ]
    subprocess.run(args, check=True)
    dst_path.parent.mkdir(exist_ok=True)
    bzl_path = Path(f"bazel-bin/spu/{libname}.so")
    shutil.copy2(bzl_path, dst_path)
    return dst_path


def build_libs():
    # pdm script command, please see the config([tool.pdm.scripts]) in pyproject.toml.
    # build so using bazel and copy to spu
    force = os.getenv("SPU_BUILD_FORCE", "false") == "true"
    _build_lib("libspu", force)
    _build_lib("libpsi", force)


def pdm_build_initialize(context):
    # pdm hook function, please refer to https://backend.pdm-project.org/hooks/
    # pdm build-config-settings: https://backend.pdm-project.org/build_config/#build-config-settings
    from packaging.tags import sys_tags, platform_tags

    context.config_settings["--plat-name"] = next(platform_tags())
    context.config_settings["--python-tag"] = next(sys_tags()).interpreter


def pdm_build_update_files(context, files: dict):
    # pdm hook function, please refer to https://backend.pdm-project.org/hooks/
    # When 'python -m build' is executed, this function will be auto called
    # and the spu/lib*.so built by bazel will be packaged into spu_*.whl
    force = os.getenv("SPU_BUILD_FORCE", "false") == "true"
    libspu = _build_lib("libspu", force)
    libpsi = _build_lib("libpsi", force)
    files[str(libspu)] = libspu
    files[str(libpsi)] = libpsi
