#! /usr/bin/env python3

# Copyright 2024 Ant Group Co., Ltd.
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
import pathlib
import shutil
import argparse
import subprocess


module_file_content = """
bazel_dep(name = "hedron_compile_commands", dev_dependency = True)
git_override(
    module_name = "hedron_compile_commands",
    # TODO: use the main branch commit if the following PR merged.
    # https://github.com/hedronvision/bazel-compile-commands-extractor/pull/219
    commit = "f5fbd4cee671d8d908f37c83abaf70fba5928fc7",
    remote = "https://github.com/hedronvision/bazel-compile-commands-extractor.git",
)
"""


def _build_file_content(content, targets):
    return f"""
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

{content}

refresh_compile_commands(
    name = "refresh_compile_commands",
    exclude_external_sources = True,
    exclude_headers = "external",
    {targets}
)

"""


def _run_shell_command_with_live_output(cmd, cwd, shell=True):
    print(cmd)
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, shell=shell
    )
    for line in p.stdout:
        print(line.decode("utf-8").rstrip())
    p.wait()
    status = p.poll()
    assert status == 0


def _rm(path: str):
    p = pathlib.Path(path)
    if os.path.isfile:
        p.unlink()
    else:
        p.rmdir()


def _backup_file(path: str):
    shutil.copy(src=path, dst=f"{path}_bak")


def _restore_file(path: str):
    backup = f"{path}_bak"
    shutil.copy(src=backup, dst=path)
    _rm(backup)


class BzlModule(object):
    def __init__(self):
        self.module_file = "MODULE.bazel"

        if os.path.isfile("BUILD.bazel"):
            self.build_file = "BUILD.bazel"
        else:
            self.build_file = "BUILD"

        self.has_build = os.path.exists(self.build_file)

    def __enter__(self):
        if self.has_build:
            _backup_file(self.build_file)
        else:
            # Create empty file
            with open(self.build_file, "w") as fp:
                pass

        _backup_file(self.module_file)
        return self

    def __exit__(self, *args):
        # Restore files
        _restore_file(self.module_file)

        if self.has_build:
            _restore_file(self.build_file)
        else:
            _rm(self.build_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="(Re)generate compile commands for Bazel project"
    )

    parser.add_argument(
        "--targets",
        metavar="bazel targets",
        type=str,
        help="bazel build targets, comma separated",
    )

    args = parser.parse_args()

    with BzlModule() as ws:
        # Add hedron_compile_commands to BzlModule
        with open(ws.module_file, "a+") as wf:
            wf.write(module_file_content)

        # Build targets
        targets = args.targets
        t_str = ""
        if targets:
            for t in targets.split(","):
                t_str += f'       "{t}": "",\n'

        if t_str:
            t_str = f"targets = {{\n{t_str}    }}"

        # Add build rule
        with open(ws.build_file, "r+") as bf:
            content = bf.read()
            bf.seek(0)

            # append load at beginning
            content = _build_file_content(content, targets=t_str)
            bf.write(content)

        # Run
        _run_shell_command_with_live_output(
            "bazelisk run -s :refresh_compile_commands", cwd=os.getcwd(), shell=True
        )