"""
warpper bazel cc_xx to modify flags.
"""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake", "configure_make")

WARNING_FLAGS = [
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wno-unused-parameter",
    "-Wnon-virtual-dtor",
] + select({
    "@bazel_tools//src/conditions:darwin": ["-Wunused-const-variable"],
    "//conditions:default": ["-Wunused-const-variable=1"],
})
DEBUG_FLAGS = ["-O0", "-g"]
RELEASE_FLAGS = ["-O2"]
FAST_FLAGS = ["-O1"]

def _spu_copts():
    return select({
        "@spulib//bazel:spu_build_as_release": RELEASE_FLAGS,
        "@spulib//bazel:spu_build_as_debug": DEBUG_FLAGS,
        "@spulib//bazel:spu_build_as_fast": FAST_FLAGS,
        "//conditions:default": FAST_FLAGS,
    }) + WARNING_FLAGS

def spu_cc_binary(
        linkopts = [],
        copts = [],
        deps = [],
        **kargs):
    cc_binary(
        linkopts = linkopts,
        copts = copts + _spu_copts(),
        deps = deps + [
            "@com_github_gperftools_gperftools//:gperftools",
        ],
        **kargs
    )

def spu_cc_library(
        linkopts = [],
        copts = [],
        deps = [],
        **kargs):
    cc_library(
        linkopts = linkopts,
        copts = _spu_copts() + copts,
        deps = deps + [
            "@com_github_gabime_spdlog//:spdlog",
        ],
        **kargs
    )

def spu_cmake_external(**attrs):
    if "generate_args" not in attrs:
        attrs["generate_args"] = ["-GNinja"]
    return cmake(**attrs)

def spu_configure_make(**attrs):
    if "args" not in attrs:
        attrs["args"] = ["-j 4"]
    return configure_make(**attrs)

def _spu_version_file_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.filename)
    ctx.actions.write(
        output = out,
        content = "__version__ = \"{}\"\n".format(ctx.attr.version),
    )
    return [DefaultInfo(files = depset([out]))]

spu_version_file = rule(
    implementation = _spu_version_file_impl,
    attrs = {
        "version": attr.string(),
        "filename": attr.string(),
    },
)

def spu_cc_test(
        linkopts = [],
        copts = [],
        deps = [],
        linkstatic = True,
        **kwargs):
    cc_test(
        # -lm for tcmalloc
        linkopts = linkopts + ["-lm"],
        copts = _spu_copts() + copts,
        deps = deps + [
            # use tcmalloc same as release bins. make them has same behavior on mem.
            "@com_github_gperftools_gperftools//:gperftools",
            "@com_google_googletest//:gtest_main",
        ],
        # static link for tcmalloc
        linkstatic = True,
        **kwargs
    )
