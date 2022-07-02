load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "psi_cpp",
    hdrs = glob([
        "include/cppcodec/**/*.hpp",
        "include/curve25519/**/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
