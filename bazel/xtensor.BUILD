load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "xtensor",
    lib_source = ":all_srcs",
    out_headers_only = True,
    deps = ["@com_github_xtensor_xtl//:xtl"],
)
