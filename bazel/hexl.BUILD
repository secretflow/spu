load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "hexl",
    cache_entries = {
        "HEXL_BENCHMARK": "OFF",
        "HEXL_TESTING": "OFF",
        "EASYLOGGINGPP_LIBRARY": "$EXT_BUILD_DEPS/easyloggingpp",
        "EASYLOGGINGPP_INCLUDEDIR": "$EXT_BUILD_DEPS/easyloggingpp/include",
    },
    lib_source = ":all_srcs",
    out_static_libs = select({
        "@spulib//bazel:spu_build_as_debug": ["libhexl_debug.a"],
        "//conditions:default": ["libhexl.a"],
    }),
    deps = [
        "@com_github_google_cpu_features//:cpu_features",
    ] + select({
        "@spulib//bazel:spu_build_as_debug": [
            "@com_github_amrayn_easyloggingpp//:easyloggingpp",
        ],
        "//conditions:default": [],
    }),
)
