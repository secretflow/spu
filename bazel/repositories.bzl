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

YACL_COMMIT_ID = "36651b8002d7ef6ac3576b4b118b2b5cf9827bf5"

def spu_deps():
    _rule_python()
    _bazel_platform()
    _upb()
    _com_github_xtensor_xtensor()
    _com_github_xtensor_xtl()
    _com_github_grpc_grpc()
    _com_github_openxla_xla()
    _com_github_pybind11_bazel()
    _com_github_pybind11()
    _com_intel_hexl()
    _com_github_amrayn_easyloggingpp()
    _com_github_emptoolkit_emp_ot()
    _com_github_facebook_zstd()
    _com_github_microsoft_seal()
    _com_github_microsoft_fourqlib()
    _com_github_eigenteam_eigen()
    _com_github_microsoft_apsi()
    _com_github_microsoft_gsl()
    _com_github_microsoft_kuku()
    _com_github_emptoolkit_emp_zk()

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

def _rule_python():
    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "8c15896f6686beb5c631a4459a3aa8392daccaab805ea899c9d14215074b60ef",
        strip_prefix = "rules_python-0.17.3",
        urls = [
            "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.17.3.tar.gz",
        ],
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
    LLVM_COMMIT = "10939d1d580b9d3c9c2f3539c6bdb39f408179c0"
    LLVM_SHA256 = "4adce5ef34c2062be0d7c5eb2a11606fa70690342e7e93327457ee2b6ad7ac72"

    OPENXLA_COMMIT = "35d98732bd9f1b753eb3de80170a29c1af878e01"
    OPENXLA_SHA256 = "e10ef5cc5c88fdcee94b8f0cf0e716698575504e255e08696dcc2e0d43eae8d2"

    TSL_COMMIT = "e1300e661b0db8aeb1063bb8f6a8182de3be7648"
    TSL_SHA256 = "a7e4258f5c8f591989ec565bc49839dd70898e586ade8e9f873150ad636266dd"

    SKYLIB_VERSION = "1.3.0"

    maybe(
        http_archive,
        name = "llvm-raw",
        build_file_content = "#empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = [
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
    )

    maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
            "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
        ],
    )

    # This is due to current status of openxla......once migration is done, we should remove this
    maybe(
        http_archive,
        name = "tsl",
        strip_prefix = "tsl-" + TSL_COMMIT,
        sha256 = TSL_SHA256,
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/tsl.patch"],
        urls = [
            "https://github.com/google/tsl/archive/{commit}.tar.gz".format(commit = TSL_COMMIT),
        ],
    )

    # We need tensorflow to handle xla->mlir hlo
    maybe(
        http_archive,
        name = "xla",
        sha256 = OPENXLA_SHA256,
        strip_prefix = "xla-" + OPENXLA_COMMIT,
        type = ".tar.gz",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/xla.patch"],
        urls = [
            "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = OPENXLA_COMMIT),
        ],
    )

def _com_github_pybind11_bazel():
    maybe(
        http_archive,
        name = "pybind11_bazel",
        sha256 = "a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d",
        strip_prefix = "pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
        urls = [
            "https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip",
        ],
    )

def _com_github_pybind11():
    maybe(
        http_archive,
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        sha256 = "111014b516b625083bef701df7880f78c2243835abdb263065b6b59b960b6bad",
        strip_prefix = "pybind11-2.10.1",
        urls = [
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.1.tar.gz",
        ],
    )

def _com_intel_hexl():
    maybe(
        http_archive,
        name = "com_intel_hexl",
        type = "tar.gz",
        strip_prefix = "hexl-1.2.3",
        sha256 = "f2cf33ee2035d12996d10b69d2f41a586b9954a29b99c70a852495cf5758878c",
        build_file = "@spulib//bazel:hexl.BUILD",
        urls = [
            "https://github.com/intel/hexl/archive/refs/tags/v1.2.3.tar.gz",
        ],
    )

def _com_github_amrayn_easyloggingpp():
    maybe(
        http_archive,
        name = "com_github_amrayn_easyloggingpp",
        type = "tar.gz",
        strip_prefix = "easyloggingpp-9.97.0",
        sha256 = "9110638e21ef02428254af8688bf9e766483db8cc2624144aa3c59006907ce22",
        build_file = "@spulib//bazel:easyloggingpp.BUILD",
        urls = [
            "https://github.com/amrayn/easyloggingpp/archive/refs/tags/v9.97.0.tar.gz",
        ],
    )

def _com_github_emptoolkit_emp_ot():
    maybe(
        http_archive,
        name = "com_github_emptoolkit_emp_ot",
        sha256 = "9c1198e04e2a081386814e9bea672fa6b4513829961c4ee150634354da609a91",
        strip_prefix = "emp-ot-0.2.2",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/emp-ot.patch"],
        urls = [
            "https://github.com/emp-toolkit/emp-ot/archive/refs/tags/0.2.2.tar.gz",
        ],
        build_file = "@spulib//bazel:emp-ot.BUILD",
    )

def _com_github_microsoft_seal():
    maybe(
        http_archive,
        name = "com_github_microsoft_seal",
        sha256 = "616653498ba8f3e0cd23abef1d451c6e161a63bd88922f43de4b3595348b5c7e",
        strip_prefix = "SEAL-4.0.0",
        type = "tar.gz",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/seal.patch"],
        urls = [
            "https://github.com/microsoft/SEAL/archive/refs/tags/v4.0.0.tar.gz",
        ],
        build_file = "@spulib//bazel:seal.BUILD",
    )

def _com_github_microsoft_fourqlib():
    maybe(
        http_archive,
        name = "com_github_microsoft_fourqlib",
        type = "zip",
        strip_prefix = "FourQlib-ff61f680505c98c98e33387962223ce0b5e620bc",
        sha256 = "59f1ebc35735217fc8c8f02c41765560ce3c5a8abd3937b0e2f4db45c49b6e73",
        build_file = "@spulib//bazel:microsoft_fourqlib.BUILD",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/fourq.patch"],
        urls = [
            "https://github.com/microsoft/FourQlib/archive/ff61f680505c98c98e33387962223ce0b5e620bc.zip",
        ],
    )

def _com_github_eigenteam_eigen():
    maybe(
        http_archive,
        name = "com_github_eigenteam_eigen",
        sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
        build_file = "@spulib//bazel:eigen.BUILD",
        strip_prefix = "eigen-3.4.0",
        urls = [
            "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
        ],
    )

def _com_github_microsoft_apsi():
    maybe(
        http_archive,
        name = "com_github_microsoft_apsi",
        sha256 = "e80fdc9489cf1223ef19e1c1e0eb36d1be83170911087099b92f27bca768d8af",
        strip_prefix = "APSI-2b61950707ca8759e31eb889081fdcd48f0a1e6c",
        type = "zip",
        urls = [
            "https://github.com/microsoft/APSI/archive/2b61950707ca8759e31eb889081fdcd48f0a1e6c.zip",
        ],
        build_file = "@spulib//bazel:microsoft_apsi.BUILD",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/apsi.patch"],
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

def _com_github_emptoolkit_emp_zk():
    maybe(
        http_archive,
        name = "com_github_emptoolkit_emp_zk",
        sha256 = "c03508b653fdc4b1251231c4c99c7240b858c598f228d66e85d61a5ba7fad39e",
        strip_prefix = "emp-zk-208195554595603c6a3f922e8318bc5b0fa67d82",
        type = "zip",
        patch_args = ["-p1"],
        patches = ["@spulib//bazel:patches/emp-zk.patch"],
        urls = [
            "https://github.com/emp-toolkit/emp-zk/archive/208195554595603c6a3f922e8318bc5b0fa67d82.zip",
        ],
        build_file = "@spulib//bazel:emp-zk.BUILD",
    )
