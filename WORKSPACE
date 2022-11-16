workspace(name = "spulib")

load("//bazel:repositories.bzl", "spu_deps")

spu_deps()

#
# yasl
# Warning: SPU relies on yasl to bring in common 3p libraries.
#          Please make sure yasl_deps are called right after spu_deps.
#
load("@yasl//bazel:repositories.bzl", "yasl_deps")

yasl_deps()

load(
    "@rules_foreign_cc//foreign_cc:repositories.bzl",
    "rules_foreign_cc_dependencies",
)

rules_foreign_cc_dependencies(
    register_built_tools = False,
    register_default_tools = False,
    register_preinstalled_tools = True,
)

load("//bazel:llvm.bzl", "llvm_setup")

llvm_setup(name = "llvm-project")

load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")

workspace()

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(
    name = "local_config_python",
    python_version = "3",
)
