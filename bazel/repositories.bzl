load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


SECRETFLOW_GIT = "https://github.com/secretflow"

YASL_COMMIT_ID  = "a1cd56d69261a0e2e4d369b0d29a4ca629ed9bc9"


def spu_deps():
    _bazel_platform()
    _com_github_xtensor_xtensor()
    _com_github_xtensor_xtl()
    _com_github_grpc_grpc()
    _com_github_tensorflow()
    _com_github_pybind11_bazel()
    _com_github_pybind11()
    _com_intel_hexl()
    _com_github_amrayn_easyloggingpp()
    _com_github_google_boringssl()
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
        name = "yasl",
        commit = YASL_COMMIT_ID,
        remote = "{}/yasl.git".format(SECRETFLOW_GIT),
    )

    # Add homebrew openmp for macOS, somehow..homebrew installs to different location on Apple Silcon/Intel macs.. so we need two rules here
    native.new_local_repository(
        name = "local_homebrew_x64",
        build_file = "@spulib//bazel:local_openmp_macos.BUILD",
        path = "/usr/local/",
    )

    native.new_local_repository(
        name = "local_homebrew_arm64",
        build_file = "@spulib//bazel:local_openmp_macos.BUILD",
        path = "/opt/homebrew/",
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

def _com_github_grpc_grpc():
    maybe(
        http_archive,
        name = "com_github_grpc_grpc",
        sha256 = "e18b16f7976aab9a36c14c38180f042bb0fd196b75c9fd6a20a2b5f934876ad6",
        strip_prefix = "grpc-1.45.2",
        type = "tar.gz",
        urls = [
            "https://github.com/grpc/grpc/archive/refs/tags/v1.45.2.tar.gz",
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

def _com_github_tensorflow():
    LLVM_COMMIT = "0538e5431afdb1fa05bdcedf70ee502ccfcd112a"
    LLVM_SHA256 = "01f168b1a8798e652a04f1faecc3d3c631ff12828b89c65503f39b0a0d6ad048"
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
    SKYLIB_VERSION = "1.2.1"
    http_archive(
        name = "bazel_skylib",
        sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
            "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
        ],
    )

    # We need tensorflow to handle xla->mlir hlo
    maybe(
        http_archive,
        name = "org_tensorflow",
        sha256 = "b5a1bb04c84b6fe1538377e5a1f649bb5d5f0b2e3625a3c526ff3a8af88633e8",
        strip_prefix = "tensorflow-2.10.0",
        patch_args = ["-p1"],
        # Fix mlir package visibility
        patches = ["@spulib//bazel:patches/tensorflow.patch"],
        type = ".tar.gz",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.10.0.tar.gz",
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
        sha256 = "eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec",
        strip_prefix = "pybind11-2.10.0",
        urls = [
            "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz",
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

# boringssl is required by grpc, we manually use a higher version.
def _com_github_google_boringssl():
    maybe(
        http_archive,
        name = "boringssl",
        sha256 = "09a9ea8b7ecdc97a7e2f128fc0fa7fcc91d781832ad19293054d3547f95fb2cd",
        strip_prefix = "boringssl-5ad11497644b75feba3163135da0909943541742",
        urls = [
            "https://github.com/google/boringssl/archive/5ad11497644b75feba3163135da0909943541742.zip",
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
