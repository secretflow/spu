load("@spulib//bazel:spu.bzl", "spu_cc_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

spu_cc_library(
    name = "cutlass",
    srcs = [],
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
