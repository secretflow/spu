#! /usr/bin/env python3

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


import os
import glob
import subprocess
import re

# Get current dir
dir_path = os.path.dirname(os.path.realpath(__file__))

# List all hlo.pb file
file_list = [f for f in glob.glob(os.path.join(dir_path, "*.hlo.pb"))]

for file in file_list:
    out = re.sub('hlo.pb$', 'mlir', file)
    stdoutdata = subprocess.getoutput(
        "../../../bazel-bin/spu/compiler/main --in={} --out={}".format(file, out)
    )
    print("Rebased {}".format(file))
