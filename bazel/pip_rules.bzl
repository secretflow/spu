# Copyright 2023 Ant Group Co., Ltd.
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

load("@rules_python//python/pip_install:requirements_parser.bzl", parse_requirements = "parse")

_BUILD_FILE_CONTENTS = """\
package(default_visibility = ["//visibility:public"])
# Ensure the `requirements.bzl` source can be accessed by stardoc, since users load() from it
exports_files(["requirements.bzl"])
"""

def _read_requirements_impl(ctx):
    requirements_file = ctx.attr.requirements
    content = parse_requirements(ctx.read(requirements_file))
    ctx.report_progress("Parsing requirements to starlark")

    ctx.file("requirements.bzl", "all_deps = [" + ",".join(['"' + str(a[1]).strip() + '"' for a in content.requirements]) + "]")

    # We need a BUILD file to load the generated requirements.bzl
    ctx.file("BUILD.bazel", _BUILD_FILE_CONTENTS)

    return

read_requirements = repository_rule(
    implementation = _read_requirements_impl,
    attrs = {
        "requirements": attr.label(),
    },
)
