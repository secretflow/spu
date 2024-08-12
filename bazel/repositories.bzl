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
    _bazel_skylib()
    _rules_cuda()
    _rules_proto_grpc()
    _bazel_platform()
    _com_github_xtensor_xtensor()
    _com_github_xtensor_xtl()
    _com_github_openxla_xla()
    _com_github_pybind11_bazel()
    _com_github_pybind11()
    _com_intel_hexl()
    _com_github_emptoolkit_emp_tool()
    _com_github_emptoolkit_emp_ot()
    _com_github_facebook_zstd()
    _com_github_eigenteam_eigen()
    _com_github_nvidia_cutlass()
    _yacl()
    _libpsi()

def _yacl():
    maybe(
        http_archive,
        name = "yacl",
        urls = [
            "https://github.com/secretflow/yacl/archive/refs/tags/0.4.5b4_nightly_20240731.tar.gz",
        ],
        strip_prefix = "yacl-0.4.5b4_nightly_20240731",
        sha256 = "952715bd56f6d9386984e9963426a1399bd2bd3702cf3efede9c82591cfab99b",
    )

def _libpsi():
    maybe(
        http_archive,
        name = "psi",
        urls = [
            "https://github.com/secretflow/psi/archive/refs/tags/v0.4.0.dev240801.tar.gz",
        ],
        strip_prefix = "psi-0.4.0.dev240801",
        sha256 = "541ad74de0cd9e6bffe348c3bc97e659fccb1f1811e612f9d8e6b1debdd7c2a0",
    )

def _rules_proto_grpc():
    http_archive(
        name = "rules_proto_grpc",
        sha256 = "2a0860a336ae836b54671cbbe0710eec17c64ef70c4c5a88ccfd47ea6e3739bd",
        strip_prefix = "rules_proto_grpc-4.6.0",
        urls = [
            "https://github.com/rules-proto-grpc/rules_proto_grpc/releases/download/4.6.0/rules_proto_grpc-4.6.0.tar.gz",
        ],
    )

def _rules_cuda():
    http_archive(
        name = "rules_cuda",
        sha256 = "b066750579f33e93e9dc55b8ee2067b525d863c1ddcf09b47a6332c39f0701fb",
        strip_prefix = "rules_cuda-v0.2.2",
        urls = ["https://github.com/bazel-contrib/rules_cuda/releases/download/v0.2.2/rules_cuda-v0.2.2.tar.gz"],
    )

def _bazel_platform():
    http_archive(
        name = "platforms",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
        ],
        sha256 = "218efe8ee736d26a3572663b374a253c012b716d8af0c07e842e82f238a0a7ee",
    )

def _com_github_facebook_zstd():
    maybe(
        http_archive,
        name = "com_github_facebook_zstd",
        build_file = "@spulib//bazel:zstd.BUILD",
        strip_prefix = "zstd-1.5.6",
        sha256 = "30f35f71c1203369dc979ecde0400ffea93c27391bfd2ac5a9715d2173d92ff7",
        type = ".tar.gz",
        urls = [
            "https://github.com/facebook/zstd/archive/refs/tags/v1.5.6.tar.gz",
        ],
    )

def _com_github_xtensor_xtensor():
    maybe(
        http_archive,
        name = "com_github_xtensor_xtensor",
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
        name = "com_github_xtensor_xtl",
        sha256 = "44fb99fbf5e56af5c43619fc8c29aa58e5fad18f3ba6e7d9c55c111b62df1fbb",
        strip_prefix = "xtl-0.7.7",
        build_file = "@spulib//bazel:xtl.BUILD",
        type = "tar.gz",
        urls = [
            "https://github.com/xtensor-stack/xtl/archive/refs/tags/0.7.7.tar.gz",
        ],
    )

def _bazel_skylib():
    maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        ],
    )

def _com_github_openxla_xla():
    OPENXLA_COMMIT = "04f2bfe797408c9efe742b89e2e4db6cf526ebb7"
    OPENXLA_SHA256 = "7e1d24737815be7607eed5f02fe7f81d97ffe358dfb7b4876f97bce8f48b3b3e"

    # We need openxla to handle xla/mhlo/stablehlo
    maybe(
        http_archive,
        name = "xla",
        sha256 = OPENXLA_SHA256,
        strip_prefix = "xla-" + OPENXLA_COMMIT,
        type = ".tar.gz",
        urls = [
            "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = OPENXLA_COMMIT),
        ],
        patch_args = ["-p1", "-l"],
        patches = ["@spulib//bazel:patches/xla-non-hermetic-python.patch"],
    )

def _com_github_pybind11_bazel():
    maybe(
        http_archive,
        name = "pybind11_bazel",
        sha256 = "dc4882b23a617575d0fd822aba88aa4a14133c3d428b5a8fb83d81d03444a475",
        strip_prefix = "pybind11_bazel-8889d39b2b925b2a47519ae09402a96f00ccf2b4",
        urls = [
            "https://github.com/pybind/pybind11_bazel/archive/8889d39b2b925b2a47519ae09402a96f00ccf2b4.zip",
        ],
    )

def _com_github_pybind11():
    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "51631e88960a8856f9c497027f55c9f2f9115cafb08c0005439838a05ba17bfc",
        strip_prefix = "pybind11-2.13.1",
        urls = [
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz",
        ],
    )

def _com_intel_hexl():
    maybe(
        http_archive,
        name = "com_intel_hexl",
        type = "tar.gz",
        strip_prefix = "hexl-1.2.5",
        sha256 = "3692e6e6183dbc49253e51e86c3e52e7affcac925f57db0949dbb4d34b558a9a",
        build_file = "@spulib//bazel:hexl.BUILD",
        urls = [
            "https://github.com/intel/hexl/archive/refs/tags/v1.2.5.tar.gz",
        ],
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/hexl.patch"],
    )

def _com_github_emptoolkit_emp_tool():
    maybe(
        http_archive,
        name = "com_github_emptoolkit_emp_tool",
        sha256 = "b9ab2380312e78020346b5d2db3d0244c7bd8098cb50f8b3620532ef491808d0",
        strip_prefix = "emp-tool-0.2.5",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = [
            "@spulib//bazel:patches/emp-tool.patch",
            "@spulib//bazel:patches/emp-tool-cmake.patch",
            "@spulib//bazel:patches/emp-tool-sse2neon.patch",
        ],
        urls = [
            "https://github.com/emp-toolkit/emp-tool/archive/refs/tags/0.2.5.tar.gz",
        ],
        build_file = "@spulib//bazel:emp-tool.BUILD",
    )

def _com_github_emptoolkit_emp_ot():
    maybe(
        http_archive,
        name = "com_github_emptoolkit_emp_ot",
        sha256 = "358036e5d18143720ee17103f8172447de23014bcfc1f8e7d5849c525ca928ac",
        strip_prefix = "emp-ot-0.2.4",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/emp-ot.patch"],
        urls = [
            "https://github.com/emp-toolkit/emp-ot/archive/refs/tags/0.2.4.tar.gz",
        ],
        build_file = "@spulib//bazel:emp-ot.BUILD",
    )

def _com_github_eigenteam_eigen():
    EIGEN_COMMIT = "66e8f38891841bf88ee976a316c0c78a52f0cee5"
    EIGEN_SHA256 = "01fcd68409c038bbcfd16394274c2bf71e2bb6dda89a2319e23fc59a2da17210"
    maybe(
        http_archive,
        name = "com_github_eigenteam_eigen",
        sha256 = EIGEN_SHA256,
        build_file = "@spulib//bazel:eigen.BUILD",
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = [
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
    )

def _com_github_nvidia_cutlass():
    maybe(
        http_archive,
        name = "com_github_nvidia_cutlass",
        strip_prefix = "cutlass-3.5.0",
        urls = [
            "https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.5.0.tar.gz",
        ],
        sha256 = "ef6af8526e3ad04f9827f35ee57eec555d09447f70a0ad0cf684a2e426ccbcb6",
        build_file = "@spulib//bazel:nvidia_cutlass.BUILD",
    )
