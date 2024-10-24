# Copyright 2022 Ant Group Co., Ltd.
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

load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "kuku",
    cache_entries = {
        "KUKU_BUILD_TESTS": "OFF",
        "KUKU_BUILD_EXAMPLES": "OFF",
        "BUILD_SHARED_LIBS": "OFF",
        "CMAKE_INSTALL_LIBDIR": "lib",
    },
    lib_source = "@com_github_microsoft_kuku//:all",
    out_include_dir = "include/Kuku-2.1",
    out_static_libs = ["libkuku-2.1.a"],
    deps = ["@com_github_microsoft_gsl//:gsl"],
)
