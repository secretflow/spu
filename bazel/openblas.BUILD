load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "openblas",
    cache_entries = {
        "NOFORTRAN": "on",
        "BUILD_WITHOUT_LAPACK": "on",
        # https://github.com/xianyi/OpenBLAS/blob/v0.3.14/USAGE.md#how-can-i-use-openblas-in-multi-threaded-applications
        # If your application is already multi-threaded, it will conflict with OpenBLAS multi-threading.
        "USE_THREAD": "off",
        "CMAKE_INSTALL_LIBDIR": "lib",
    },
    lib_source = ":all_srcs",
    out_lib_dir = "lib",
    out_static_libs = [
        "libopenblas.a",
    ],
)
