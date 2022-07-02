load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "zlib",
    srcs = glob([
        "*.c",
        "*.h",
    ]),
    hdrs = [
        "zconf.h",
        "zlib.h",
    ],
    copts = [
        "-Wno-implicit-function-declaration",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
