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
    name = "apsi",
    cache_entries = {
        "APSI_USE_LOG4CPLUS": "OFF",
        "CMAKE_INSTALL_LIBDIR": "lib",
    },
    copts = [
        "-DAPSI_DISABLE_JSON",
        "-I$EXT_BUILD_DEPS/gsl/include",
    ],
    lib_source = "@com_github_microsoft_apsi//:all",
    out_include_dir = "include/APSI-0.8",
    out_static_libs = ["libapsi-0.8.a"],
    deps = [
        "@com_github_facebook_zstd//:zstd",
        "@com_github_microsoft_gsl//:gsl",
        "@com_github_microsoft_kuku//:kuku",
        "@com_github_microsoft_seal//:seal",
    ],
)
