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

set +e

if [ "$#" -ne 1 ]; then
    echo "Missing benchmark name"
fi

COLOR_GREEN="\033[92m"
COLOR_END="\033[0m"

echo -e "${COLOR_GREEN}Cleanup old docker${COLOR_END}"
docker rm -f spu-build

echo -e "${COLOR_GREEN}Build spu-build${COLOR_END}"
docker run --name spu-build --mount type=bind,source="$(pwd)",target=/home/admin/dev/ secretflow/spu-ci:0.6 \
    sh -c "cd /home/admin/dev && \
            python3 -m pip install -U pip && \
            python3 -m pip install -r requirements.txt && \
            bazel build //benchmark/... //examples/python/... -c opt --ui_event_filters=-info,-debug,-warning"

docker commit spu-build spu-build:v1

echo -e "${COLOR_GREEN}Startup docker compose${COLOR_END}"
docker-compose -f .circleci/benchmark.yml up -d
sleep 10

echo -e "${COLOR_GREEN}Run benchmark${COLOR_END}"
docker run --rm --mount type=bind,source="$(pwd)",target=/home/admin/dev/ --network nn-benchmark spu-build:v1 \
    sh -c "cd /home/admin/dev && bash benchmark/run_bench.sh $@" | tee benchmark_results.log

echo -e "${COLOR_GREEN}Shutdown docker compose${COLOR_END}"
docker-compose -f .circleci/benchmark.yml down

echo -e "${COLOR_GREEN}Benchmark output${COLOR_END}"
cat benchmark_results.log
