load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "emp-ot",
    cache_entries = {
        "CMAKE_FOLDER": "$EXT_BUILD_DEPS/emp-tool",
        "EMP-TOOL_INCLUDE_DIR": "$EXT_BUILD_DEPS/emp-tool/include",
        "EMP-TOOL_LIBRARY": "$EXT_BUILD_DEPS/emp-tool/lib",
        "OPENSSL_ROOT_DIR": "$EXT_BUILD_DEPS/openssl",
        "BUILD_TESTING": "OFF",
    },
    lib_source = ":all_srcs",
    out_headers_only = True,
    deps = [
        "@com_github_emptoolkit_emp_tool//:emp-tool",
        "@com_github_openssl_openssl//:openssl",
    ],
)
