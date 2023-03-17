#! /bin/bash
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

script_dir=$(realpath $(dirname $0))
work_dir=${script_dir}/../..

echo "OUTPUT_DIR: " $script_dir
cd $work_dir

TEMP=$(getopt -o r:,d:,v:,p,:d,r:,m:,o: -l protocol:,remote:,remote_dir:,remote_python_env:,parties:,delay:,rate:,mode:,output: -n "$0" -- "$@" )

if [ $? != 0 ]; then echo "Arg parse failed"; fi

eval set -- "${TEMP}"

optstr="--"

echo ${TEMP}
while true; do
  case "$1" in
    --protocol) protocol=$2; optstr="${optstr} --protocol=$protocol"; shift 2
      ;;
    --remote) remote=$2; shift 2
      ;;
    --remote_dir) remote_dir=$2; shift 2
      ;;
    --remote_python_env) remote_python_env=$2; shift 2
      ;;
    --parties) parties=$2; optstr="${optstr} --parties=$parties"; shift 2
      ;;
    --delay) delay=$2; shift 2
      ;;
    --rate) rate=$2; shift 2
      ;;
    --mode) mode=$2; optstr="${optstr} --mode=$mode"; shift 2
      ;;
    --output) output=$2; shift 2
      ;;
    --) shift; break
      ;;
    *) echo "Usage: \
${0} --remote ip:port --remote_dir /dir_to_ppu \
--remote_python_env /dir_to_python_activate --parties local_ip:addr,remote_ip:addr,... \
 --delay xxsec --rate xxbit --mode xxx --output xxx.json \
    " ; exit 1
    ;;
  esac
done

for arg do
  optstr="$optstr --$arg"
done


if [ -z "$mode" ] || [[ $mode =~ "standalone" ]]; then
  if [ -z $output ]; then
    output=standalone.json
  fi
  bazel run -c opt //libspu/mpc/tools:benchmark ${optstr} \
          --benchmark_out=${script_dir}/$output \
         --benchmark_out_format=json
fi

if [[ "${mode}" =~ "mparty" ]]; then
        if [ -z $remote ]; then
    if [ -z $output ]; then
      output=LAN.json
    fi
  bazel run -c opt //libspu/mpc/tools:benchmark ${optstr} --rank=1>/dev/null 2>&1 &
  bazel run -c opt //libspu/mpc/tools:benchmark ${optstr} --rank=2>/dev/null 2>&1 &
  bazel run -c opt //libspu/mpc/tools:benchmark ${optstr} \
                --benchmark_out=${script_dir}/$output --rank=0 \
                --benchmark_out_format=json
  else
    if [ -z $output ]; then
      output=WAN_${rate}_${delay}.json
    fi
    ssh ${remote} "cd ${remote_dir}; source ${remote_python_env}; \
      bazel run -c opt //libspu/mpc/tools:benchmark \
      ${optstr} --rank=1 >/dev/null 2>&1" &

    if [ ${protocol} == 'aby3' || -z ${protocol} ]; then
      ssh ${remote} "cd ${remote_dir}; source ${remote_python_env}; \
        bazel run -c opt //libspu/mpc/tools:benchmark \
        ${optstr} --rank=2 >/dev/null 2>&1" &
    fi
    tc qdisc del dev eth0 root
    if [ -n ${rate} ]; then
      tc qdisc add dev eth0 root handle 1: tbf \
        rate ${rate} burst 256kb latency 800ms
    fi
    if [ -n ${delay} ]; then
      tc qdisc add dev eth0 parent 1:1 handle 10: \
        netem delay ${delay} limit 8000
    fi

    bazel run -c opt //libspu/mpc/tools:benchmark ${optstr} \
      --rank=0 --benchmark_out="${script_dir}/$output" \
      --benchmark_out_format=json

    tc qdisc del dev eth0 root
  fi

fi
