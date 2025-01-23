# Copyright 2021 Ant Group Co., Ltd.
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

"""
warpper bazel cc_xx to modify flags.
"""

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@spu_pip//:requirements.bzl", pip_dep = "all_requirements")

def spu_py_binary(
        deps = [],
        **kwargs):
    py_binary(
        deps = deps + pip_dep,
        **kwargs
    )

def spu_py_library(
        deps = [],
        **kwargs):
    py_library(
        deps = deps + pip_dep,
        **kwargs
    )

def spu_py_test(
        deps = [],
        **kwargs):
    py_test(
        deps = deps + pip_dep,
        **kwargs
    )
