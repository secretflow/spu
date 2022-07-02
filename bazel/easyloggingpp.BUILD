load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "easyloggingpp",
    cache_entries = {
        "build_static_lib": "ON",
    },
    lib_source = ":all_srcs",
    out_static_libs = [
        "libeasyloggingpp.a",
    ],
)
