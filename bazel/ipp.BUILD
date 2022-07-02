load("@spulib//bazel:spu.bzl", "spu_cmake_external")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

spu_cmake_external(
    name = "ipp",
    cache_entries = {
        "ARCH": "intel64",
        "OPENSSL_INCLUDE_DIR": "$EXT_BUILD_DEPS/openssl/include",
        "OPENSSL_LIBRARIES": "$EXT_BUILD_DEPS/openssl/lib",
        "OPENSSL_ROOT_DIR": "$EXT_BUILD_DEPS/openssl",
        "CMAKE_BUILD_TYPE": "Release",
    },
    lib_source = ":all_srcs",
    out_static_libs = [
        "intel64/libippcp.a",
        "intel64/libcrypto_mb.a",
    ],
    deps = [
        "@com_github_openssl_openssl//:openssl",
    ],
)
