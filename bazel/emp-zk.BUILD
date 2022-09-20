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
    name = "emp-zk",
    cache_entries = {
        "CMAKE_FOLDER": "$EXT_BUILD_DEPS/emp-tool",
        "EMP-TOOL_INCLUDE_DIR": "$EXT_BUILD_DEPS/emp-tool/include",
        "EMP-TOOL_LIBRARY": "$EXT_BUILD_DEPS/emp-tool/lib",
        "EMP-OT_INCLUDE_DIR": "$EXT_BUILD_DEPS/emp-ot/include",
        "EMP-OT_LIBRARY": "$EXT_BUILD_DEPS/emp-ot/lib",
        "OPENSSL_ROOT_DIR": "$EXT_BUILD_DEPS/openssl",
        "BUILD_TESTING": "OFF",
    },
    lib_source = ":all_srcs",
    out_headers_only = True,
    deps = [
        "@com_github_emptoolkit_emp_ot//:emp-ot",
        "@com_github_emptoolkit_emp_tool//:emp-tool",
        "@com_github_openssl_openssl//:openssl",
    ],
)
