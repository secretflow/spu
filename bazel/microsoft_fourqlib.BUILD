load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

make(
    name = "fourqlib",
    args = select({
        "@bazel_tools//src/conditions:darwin_x86_64": [
            "ARCH=x64",
            "AVX2=TRUE",
            "ASM=FALSE",
        ],
        "@bazel_tools//src/conditions:linux_aarch64": [
            "ARCH=ARM64",
        ],
        "@bazel_tools//src/conditions:darwin_arm64": [
            "ARCH=ARM64",
        ],
        "//conditions:default": [
            "ARCH=x64",
            "AVX2=TRUE",
            "ASM=TRUE",
        ],
    }),
    defines = [
        "__LINUX__",
    ] + select({
        "@platforms//cpu:x86_64": [
            "_AMD64_",
        ],
        "//conditions:default": [
            "_ARM64_",
        ],
    }),
    env = select({
        "@bazel_tools//src/conditions:darwin": {
            "AR": "ar",
        },
        "//conditions:default": {},
    }),
    lib_source = ":all_srcs",
    out_static_libs = ["libfourq.a"],
    targets = ["install"],
    tool_prefix = "export BUILD_TMPDIR=$BUILD_TMPDIR/FourQ_64bit_and_portable &&",
)
