# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

# The subset of LLVM targets that SPU cares about.
_LLVM_TARGETS = [
    "NVPTX",  # Somehow mlir tests need this....
    "X86",
    "AArch64",
]

def llvm_setup(name):
    # Disable terminfo and zlib that are bundled with LLVM.
    llvm_disable_optional_support_deps()

    # Build @llvm-project from @llvm-raw using overlays.
    llvm_configure(
        name = name,
        repo_mapping = {"@python_runtime": "@local_config_python"},
        targets = _LLVM_TARGETS,
    )
