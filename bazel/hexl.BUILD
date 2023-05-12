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
    name = "all_srcs",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "hexl",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CpuFeatures_DIR": "$EXT_BUILD_DEPS/cpu_features/lib/cmake/CpuFeatures/",
        "HEXL_BENCHMARK": "OFF",
        "HEXL_TESTING": "OFF",
        "CMAKE_INSTALL_LIBDIR": "lib",
    },
    lib_source = ":all_srcs",
    out_data_dirs = ["lib/cmake"],
    out_static_libs = ["libhexl.a"],
    deps = [
        "@com_github_google_cpu_features//:cpu_features",
    ],
)
