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
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

SECRETFLOW_GIT = "https://github.com/secretflow"

YACL_COMMIT_ID = "e42508bc288a5fee70cb44ae6fce555483dc5265"

def spu_deps():
    _bazel_platform()
    _upb()
    _com_github_xtensor_xtensor()
    _com_github_xtensor_xtl()
    _com_github_grpc_grpc()
    _com_github_openxla_xla()
    _com_github_pybind11_bazel()
    _com_github_pybind11()
    _com_intel_hexl()
    _com_github_emptoolkit_emp_tool()
    _com_github_emptoolkit_emp_ot()
    _com_github_emptoolkit_emp_zk()
    _com_github_facebook_zstd()
    _com_github_microsoft_seal()
    _com_github_eigenteam_eigen()
    _com_github_microsoft_apsi()
    _com_github_microsoft_gsl()
    _com_github_microsoft_kuku()
    _com_google_flatbuffers()

    maybe(
        git_repository,
        name = "yacl",
        commit = YACL_COMMIT_ID,
        remote = "{}/yacl.git".format(SECRETFLOW_GIT),
    )

    # Add homebrew openmp for macOS, somehow..homebrew installs to different location on Apple Silcon/Intel macs.. so we need two rules here
    native.new_local_repository(
        name = "local_homebrew_x64",
        build_file = "@spulib//bazel:local_openmp_macos.BUILD",
        path = "/usr/local/opt/libomp",
    )

    native.new_local_repository(
        name = "local_homebrew_arm64",
        build_file = "@spulib//bazel:local_openmp_macos.BUILD",
        path = "/opt/homebrew/opt/libomp/",
    )

def _bazel_platform():
    http_archive(
        name = "platforms",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
        ],
        sha256 = "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
    )

def _com_github_facebook_zstd():
    maybe(
        http_archive,
        name = "com_github_facebook_zstd",
        build_file = "@spulib//bazel:zstd.BUILD",
        strip_prefix = "zstd-1.5.0",
        sha256 = "5194fbfa781fcf45b98c5e849651aa7b3b0a008c6b72d4a0db760f3002291e94",
        type = ".tar.gz",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.0/zstd-1.5.0.tar.gz",
        ],
    )

def _upb():
    maybe(
        http_archive,
        name = "upb",
        sha256 = "017a7e8e4e842d01dba5dc8aa316323eee080cd1b75986a7d1f94d87220e6502",
        strip_prefix = "upb-e4635f223e7d36dfbea3b722a4ca4807a7e882e2",
        urls = [
            "https://storage.googleapis.com/grpc-bazel-mirror/github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
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
        sha256 = "37738aa0865350b39f048e638735c05d78b5331073b6329693e8b8f0902df713",
        strip_prefix = "xtensor-0.24.0",
        build_file = "@spulib//bazel:xtensor.BUILD",
        type = "tar.gz",
        urls = [
            "https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.24.0.tar.gz",
        ],
    )

def _com_github_xtensor_xtl():
    maybe(
        http_archive,
        name = "com_github_xtensor_xtl",
        sha256 = "f4a81e3c9ca9ddb42bd4373967d4859ecfdca1aba60b9fa6ced6c84d8b9824ff",
        strip_prefix = "xtl-0.7.3",
        build_file = "@spulib//bazel:xtl.BUILD",
        type = "tar.gz",
        urls = [
            "https://github.com/xtensor-stack/xtl/archive/refs/tags/0.7.3.tar.gz",
        ],
    )

def _com_github_openxla_xla():
    OPENXLA_COMMIT = "7949c6d44bbc6ae5d1c36a8cdcd27520de509d74"
    OPENXLA_SHA256 = "39951250ae922fdf22f370ebb63f3a8d1b54fcfa822a324538be12238c899fde"

    SKYLIB_VERSION = "1.3.0"

    maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
            "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
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
        sha256 = "6426567481ee345eb48661e7db86adc053881cb4dd39fbf527c8986316b682b9",
        strip_prefix = "pybind11_bazel-fc56ce8a8b51e3dd941139d329b63ccfea1d304b",
        urls = [
            "https://github.com/pybind/pybind11_bazel/archive/fc56ce8a8b51e3dd941139d329b63ccfea1d304b.zip",
        ],
    )

def _com_github_pybind11():
    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "5d8c4c5dda428d3a944ba3d2a5212cb988c2fae4670d58075a5a49075a6ca315",
        strip_prefix = "pybind11-2.10.3",
        urls = [
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.tar.gz",
        ],
    )

def _com_intel_hexl():
    maybe(
        http_archive,
        name = "com_intel_hexl",
        type = "tar.gz",
        strip_prefix = "hexl-1.2.4",
        sha256 = "f91dd158e3faa8795035f618286cdef7dacbf488fd0ba4f44409880997b4549f",
        build_file = "@spulib//bazel:hexl.BUILD",
        urls = [
            "https://github.com/intel/hexl/archive/refs/tags/v1.2.4.tar.gz",
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

def _com_github_emptoolkit_emp_zk():
    maybe(
        http_archive,
        name = "com_github_emptoolkit_emp_zk",
        sha256 = "e02e6abc6ee14ca0e69e6f5f0efe24cab7da1bc905fc7c86a3e5a529114e489a",
        strip_prefix = "emp-zk-0.2.1",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/emp-zk.patch"],
        urls = [
            "https://github.com/emp-toolkit/emp-zk/archive/refs/tags/0.2.1.tar.gz",
        ],
        build_file = "@spulib//bazel:emp-zk.BUILD",
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
    maybe(
        http_archive,
        name = "com_github_eigenteam_eigen",
        sha256 = "c1b115c153c27c02112a0ecbf1661494295d9dcff6427632113f2e4af9f3174d",
        build_file = "@spulib//bazel:eigen.BUILD",
        strip_prefix = "eigen-3.4",
        urls = [
            "https://gitlab.com/libeigen/eigen/-/archive/3.4/eigen-3.4.tar.gz",
        ],
    )

def _com_github_microsoft_apsi():
    maybe(
        http_archive,
        name = "com_github_microsoft_apsi",
        sha256 = "82c0f9329c79222675109d4a3682d204acd3ea9a724bcd98fa58eabe53851333",
        strip_prefix = "APSI-0.11.0",
        urls = [
            "https://github.com/microsoft/APSI/archive/refs/tags/v0.11.0.tar.gz",
        ],
        build_file = "@spulib//bazel:microsoft_apsi.BUILD",
        patch_args = ["-p1"],
        patches = [
            "@spulib//bazel:patches/apsi.patch",
            "@spulib//bazel:patches/apsi-gen.patch",
            "@spulib//bazel:patches/apsi_bin_bundle.patch",
        ],
    )

def _com_github_microsoft_gsl():
    maybe(
        http_archive,
        name = "com_github_microsoft_gsl",
        sha256 = "f0e32cb10654fea91ad56bde89170d78cfbf4363ee0b01d8f097de2ba49f6ce9",
        strip_prefix = "GSL-4.0.0",
        type = "tar.gz",
        urls = [
            "https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.tar.gz",
        ],
        build_file = "@spulib//bazel:microsoft_gsl.BUILD",
    )

def _com_github_microsoft_kuku():
    maybe(
        http_archive,
        name = "com_github_microsoft_kuku",
        sha256 = "96ed5fad82ea8c8a8bb82f6eaf0b5dce744c0c2566b4baa11d8f5443ad1f83b7",
        strip_prefix = "Kuku-2.1.0",
        type = "tar.gz",
        urls = [
            "https://github.com/microsoft/Kuku/archive/refs/tags/v2.1.0.tar.gz",
        ],
        build_file = "@spulib//bazel:microsoft_kuku.BUILD",
    )

def _com_google_flatbuffers():
    maybe(
        http_archive,
        name = "com_google_flatbuffers",
        sha256 = "8aff985da30aaab37edf8e5b02fda33ed4cbdd962699a8e2af98fdef306f4e4d",
        strip_prefix = "flatbuffers-23.3.3",
        urls = [
            "https://github.com/google/flatbuffers/archive/refs/tags/v23.3.3.tar.gz",
        ],
    )
