workspace(name = "spulib")

load("//bazel:repositories.bzl", "spu_deps")

spu_deps()

#
# yacl
# Warning: SPU relies on yacl to bring in common 3p libraries.
#          Please make sure yacl_deps are called right after spu_deps.
#
load("@yacl//bazel:repositories.bzl", "yacl_deps")

yacl_deps()

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

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(
    name = "local_config_python",
    python_version = "3",
)
