#!/usr/bin/env python3

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

import argparse
import io
import os
import sys

IMPORT_KEY = '# DO-NOT-EDIT:ADD_IMPORT'
LIST_KEY = '# DO-NOT-EDIT:EOL'
CPP_DISPATCH_KEY = '// DO-NOT-EDIT: Add_DISPATCH_CODE'


# Adds the implementation of the new check.
def write_python_implementation(intrinsic_path, intrinsic_name):
    filename = os.path.join(intrinsic_path, intrinsic_name + "_impl") + ".py"
    print("Creating %s..." % filename)

    # Read template
    with io.open(
        os.path.join(intrinsic_path, "intrinsic_impl_template.txt"), "r"
    ) as template:
        content = template.read()
        content = content.replace("{%NAME}", intrinsic_name)
        with io.open(filename, "w", encoding="utf8", newline="\n") as f:
            f.write(content)


def update_cpp_file(intrinsic_path, intrinsic_name):
    filename = os.path.join(
        intrinsic_path,
        "..",
        "..",
        "libspu",
        "device",
        "pphlo",
        "pphlo_intrinsic_executor.cc",
    )
    print("Updating %s..." % filename)

    dispatch_code = f'''
  if (name == "{intrinsic_name}") {{
      SPU_THROW("Missing implementation for {{}}", name);
  }}
  {CPP_DISPATCH_KEY}
            '''

    with io.open(filename, "r+") as f:
        content = f.read()
        content = content.replace(CPP_DISPATCH_KEY, dispatch_code)
        f.seek(0)
        f.write(content)


# Update module __init__.py
def adapt_module(module_path, check_name):
    init_file = os.path.join(module_path, "__init__.py")
    print(f'Updating "{init_file}"...')
    IMPORT_KEY = '# DO-NOT-EDIT:ADD_IMPORT'
    LIST_KEY = '# DO-NOT-EDIT:EOL'
    with io.open(init_file, "r+", encoding="utf8") as f:
        content = f.read()
        content = content.replace(
            IMPORT_KEY,
            f'from .{check_name}_impl import {check_name}\n{IMPORT_KEY}',
        )
        content = content.replace(
            LIST_KEY,
            f'"{check_name}",\n\t{LIST_KEY}',
        )
        f.seek(0)
        f.write(content)


def adapt_build(module_path, check_name):
    build_file = os.path.join(module_path, "BUILD.bazel")
    print(f'Updating "{build_file}"...')

    with io.open(build_file, "r+", encoding="utf8") as f:
        content = f.read()
        content = content.replace(
            LIST_KEY,
            f'":{check_name}",\n\t\t{IMPORT_KEY}',
        )
        content = (
            content
            + f"""
spu_py_library(
    name = "{check_name}",
    srcs = [
        "{check_name}_impl.py",
    ],
    visibility = [
        "//visibility:private",
    ],
)
        """
        )
        f.seek(0)
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "intrinsic", nargs="?", help="name of new intrinsic to add (e.g. my_magic_fcn)"
    )
    args = parser.parse_args()

    if not args.intrinsic:
        print("Intrinsic must be specified.")
        parser.print_usage()
        return

    intrinsic_name = args.intrinsic
    intrinsic_path = os.path.dirname(sys.argv[0])

    write_python_implementation(intrinsic_path, intrinsic_name)
    adapt_module(intrinsic_path, intrinsic_name)
    adapt_build(intrinsic_path, intrinsic_name)
    update_cpp_file(intrinsic_path, intrinsic_name)
    print("Done. Now it's your turn!")


if __name__ == "__main__":
    main()
