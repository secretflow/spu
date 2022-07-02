load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "seal",
    cache_entries = {
        "SEAL_USE_MSGSL": "OFF",
        "SEAL_BUILD_DEPS": "OFF",
        "SEAL_USE_ZSTD": "OFF",
        "SEAL_USE_ZLIB": "OFF",
        "CMAKE_INSTALL_LIBDIR": "lib",
    },
    lib_source = "@com_github_microsoft_seal//:all",
    out_include_dir = "include/SEAL-3.6",
    out_static_libs = ["libseal-3.6.a"],
    deps = ["@zlib"],
)
