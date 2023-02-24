#!/usr/bin/bash
#
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
#

rm -rf test-results
mkdir -p test-results

# renaming junit xml file to satisfy ci's requirement
for path in $(find bazel-testlogs/ -name "test.xml"); do
    dir_name=$(dirname ${path})
    file_name=$(basename ${path})
    path_md5=$(echo ${path} | md5sum | cut -f1 -d ' ')
    target="test-results/TEST-${path_md5}.xml"
    echo "mv $path to ${target} ..."
    mv ${path} ${target}
done
