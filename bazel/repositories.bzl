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
    _rules_cuda()
    _rules_proto_grpc()
    _bazel_platform()
    _com_github_xtensor_xtensor()
    _com_github_xtensor_xtl()
    _com_github_grpc_grpc()
    _com_github_openxla_xla()
    _com_github_pybind11_bazel()
    _com_github_pybind11()
    _com_intel_hexl()
    _com_github_emptoolkit_emp_tool()
    _com_github_emptoolkit_emp_ot()
    _com_github_facebook_zstd()
    _com_github_microsoft_seal()
    _com_github_eigenteam_eigen()
    _com_github_nvidia_cutlass()
    _yacl()
    _libpsi()

def _yacl():
    maybe(
        http_archive,
        name = "yacl",
        urls = [
            "https://github.com/secretflow/yacl/archive/c1721d4671be03adf46b2094f12a6e895eb02f8b.tar.gz",
        ],
        strip_prefix = "yacl-c1721d4671be03adf46b2094f12a6e895eb02f8b",
        sha256 = "06c72b715e7132a742049c7f491387baa4e6be60dab063093e59516376bc8d14",
    )

def _libpsi():
    maybe(
        http_archive,
        name = "psi",
        urls = [
            "https://github.com/secretflow/psi/archive/cee136c0fbbfdc6041931c303035c09e389947fa.tar.gz",
        ],
        strip_prefix = "psi-cee136c0fbbfdc6041931c303035c09e389947fa",
        sha256 = "5e1d2ef9708af5dee6a2018dabc8c53412ec389b7cddc151399e8c5065f5374d",
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
        sha256 = "2f8c8c8c85f727bec4423efecec12d3b751cb0a98bda99f0f9d351608a23b858",
        strip_prefix = "rules_cuda-v0.2.1",
        urls = ["https://github.com/bazel-contrib/rules_cuda/releases/download/v0.2.1/rules_cuda-v0.2.1.tar.gz"],
    )

def _bazel_platform():
    http_archive(
        name = "platforms",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.8/platforms-0.0.8.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.8/platforms-0.0.8.tar.gz",
        ],
        sha256 = "8150406605389ececb6da07cbcb509d5637a3ab9a24bc69b1101531367d89d74",
    )

def _com_github_facebook_zstd():
    maybe(
        http_archive,
        name = "com_github_facebook_zstd",
        build_file = "@spulib//bazel:zstd.BUILD",
        strip_prefix = "zstd-1.5.5",
        sha256 = "98e9c3d949d1b924e28e01eccb7deed865eefebf25c2f21c702e5cd5b63b85e1",
        type = ".tar.gz",
        urls = [
            "https://github.com/facebook/zstd/archive/refs/tags/v1.5.5.tar.gz",
        ],
    )

def _com_github_grpc_grpc():
    maybe(
        http_archive,
        name = "com_github_grpc_grpc",
        sha256 = "7f42363711eb483a0501239fd5522467b31d8fe98d70d7867c6ca7b52440d828",
        strip_prefix = "grpc-1.51.0",
        type = "tar.gz",
        patch_args = ["-p1"],
        # Set grpc to use local go toolchain
        patches = ["@spulib//bazel:patches/grpc.patch"],
        urls = [
            "https://github.com/grpc/grpc/archive/refs/tags/v1.51.0.tar.gz",
        ],
    )

def _com_github_xtensor_xtensor():
    maybe(
        http_archive,
        name = "com_github_xtensor_xtensor",
        sha256 = "0fbbd524dde2199b731b6af99b16063780de6cf1d0d6cb1f3f4d4ceb318f3106",
        strip_prefix = "xtensor-0.24.7",
        build_file = "@spulib//bazel:xtensor.BUILD",
        type = "tar.gz",
        urls = [
            "https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.24.7.tar.gz",
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

def _com_github_openxla_xla():
    OPENXLA_COMMIT = "fa9331a7e557b4ec1381f84cbbf7401a8f41ac66"
    OPENXLA_SHA256 = "d19c570d434002b7b0490327d407fc7cf2b18633f4a2d3b1bb44f3f0e4b36533"

    maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz",
        ],
    )

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
        sha256 = "d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c",
        strip_prefix = "pybind11-2.11.1",
        urls = [
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz",
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

def _com_github_microsoft_seal():
    maybe(
        http_archive,
        name = "com_github_microsoft_seal",
        sha256 = "af9bf0f0daccda2a8b7f344f13a5692e0ee6a45fea88478b2b90c35648bf2672",
        strip_prefix = "SEAL-4.1.1",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/seal.patch"],
        urls = [
            "https://github.com/microsoft/SEAL/archive/refs/tags/v4.1.1.tar.gz",
        ],
        build_file = "@spulib//bazel:seal.BUILD",
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
        strip_prefix = "cutlass-3.4.0",
        urls = [
            "https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.4.0.tar.gz",
        ],
        sha256 = "49f4b854acc2a520126ceefe4f701cfe8c2b039045873e311b1f10a8ca5d5de1",
        build_file = "@spulib//bazel:nvidia_cutlass.BUILD",
    )
