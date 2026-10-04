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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def spu_deps():
    _com_github_xtensor_xtensor()
    _com_github_xtensor_xtl()

def _com_github_xtensor_xtensor():
    maybe(
        http_archive,
        name = "xtensor",
        sha256 = "32d5d9fd23998c57e746c375a544edf544b74f0a18ad6bc3c38cbba968d5e6c7",
        strip_prefix = "xtensor-0.25.0",
        build_file = "@spulib//bazel:xtensor.BUILD",
        type = "tar.gz",
        urls = [
            "https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.25.0.tar.gz",
        ],
    )

def _com_github_xtensor_xtl():
    maybe(
        http_archive,
        name = "xtl",
        sha256 = "8fb38d6a5856aab5740d2ccb3d791d289f648d4cc506b94a1338fe5fce100c11",
        strip_prefix = "xtl-0.8.2",
        build_file = "@spulib//bazel:xtl.BUILD",
        type = "tar.gz",
        urls = [
            "https://github.com/xtensor-stack/xtl/archive/refs/tags/0.8.2.tar.gz",
        ],
    )
