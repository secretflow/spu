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

load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

make(
    name = "fourqlib",
    args = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [
            "ARCH=x64",
            "AVX2=TRUE",
            "ASM=FALSE",
        ],
        "@platforms//cpu:aarch64": [
            "ARCH=ARM64",
        ],
        "//conditions:default": [
            "ARCH=x64",
            "AVX2=TRUE",
            "ASM=TRUE",
        ],
    }),
    defines = [
        "__LINUX__",
    ] + select({
        "@platforms//cpu:x86_64": [
            "_AMD64_",
        ],
        "//conditions:default": [
            "_ARM64_",
        ],
    }),
    env = select({
        "@bazel_tools//src/conditions:darwin": {
            "AR": "ar",
        },
        "//conditions:default": {},
    }),
    lib_source = ":all_srcs",
    out_static_libs = ["libfourq.a"],
    targets = ["install"],
    tool_prefix = "cd $$BUILD_TMPDIR/FourQ_64bit_and_portable &&",
)
