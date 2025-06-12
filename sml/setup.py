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

from datetime import date
import os
import re
import subprocess
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))


def get_commit_id() -> str:
    commit_id = (
        subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    )
    dirty = subprocess.check_output(['git', 'diff', '--stat']).decode('ascii').strip()

    if dirty:
        commit_id = f"{commit_id}-dirty"

    return commit_id


def complete_version_file(*filepath):
    today = date.today()
    dstr = today.strftime("%Y%m%d")
    with open(os.path.join(".", *filepath), "r") as fp:
        content = fp.read()

    content = content.replace("$$DATE$$", dstr)
    try:
        content = content.replace("$$COMMIT_ID$$", get_commit_id())
    except:
        pass

    with open(os.path.join(".", *filepath), "w+") as fp:
        fp.write(content)


def find_version(*filepath):
    complete_version_file(*filepath)
    # Extract version information from filepath
    with open(os.path.join(".", *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        print("Unable to find version string.")
        exit(-1)


def read_requirements():
    requirements = []
    with open("./requirements.txt") as file:
        requirements = file.read().splitlines()
    print("Requirements: ", requirements)
    return requirements


if __name__ == "__main__":
    if os.getcwd() != this_directory:
        print("You must run setup.py from the `sml` dir")
        exit(-1)

    pkg_exclude_list = [
        "emulations",
        "emulations.*",
        "tests",
        "tests.*",
    ]

    install_requires = read_requirements()

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name="sml",
        version=find_version("version.py"),
        author="SecretFlow Team",
        author_email="secretflow-contact@service.alipay.com",
        description="Secretflow Secure Machine Learning",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/secretflow/spu",
        packages=find_packages(exclude=pkg_exclude_list),
        install_requires=install_requires,
        extras_require={"dev": ["pylint"]},
        python_requires=">=3.10, <3.12",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        license="Apache 2.0",
    )
