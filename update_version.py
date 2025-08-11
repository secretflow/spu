#! /usr/bin/env python3

# Copyright 2025 Ant Group Co., Ltd.
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
import os
import re

SPU_VERSION_PREFIX = "0.9.4.dev"


def get_today_tag():
    import datetime

    return datetime.date.today().strftime('%Y%m%d')


def update_first_matched_pattern_in_file(file_path, pattern, replacement):
    if not os.path.exists(file_path):
        print(f"[WARN] Not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        new_content = re.sub(pattern, replacement, content, count=1)

        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(new_content)


def main():
    parser = argparse.ArgumentParser(description="Update SPU release version")

    parser.add_argument(
        "--version",
        metavar="bazel module version",
        type=str,
        help="bazel module version",
        default=f'{SPU_VERSION_PREFIX}{get_today_tag()}',
    )

    args = parser.parse_args()
    new_version = args.version

    version_pattern1 = 'version = \"{}\"'
    update_first_matched_pattern_in_file(
        "MODULE.bazel",
        version_pattern1.format('[^"]*'),
        version_pattern1.format(new_version),
    )
    update_first_matched_pattern_in_file(
        "src/MODULE.bazel",
        version_pattern1.format('[^"]*'),
        version_pattern1.format(new_version),
    )

    version_pattern2 = 'SPU_VERSION = \"{}\"'
    update_first_matched_pattern_in_file(
        "version.bzl",
        version_pattern2.format('[^"]*'),
        version_pattern2.format(new_version),
    )

    version_pattern3 = '#define SPU_VERSION \"{}\"'
    update_first_matched_pattern_in_file(
        "src/libspu/version.h",
        version_pattern3.format('[^"]*'),
        version_pattern3.format(new_version),
    )


if __name__ == "__main__":
    main()
