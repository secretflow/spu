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

import argparse
import json
import multiprocess
import sys

import grpc
import jax

import spu.utils.distributed as ppd
from spu.utils import distributed_pb2_grpc

parser = argparse.ArgumentParser(description='SPU node service.')
parser.add_argument(
    "-c", "--config", default="examples/python/conf/3pc.json", help="the config"
)
subparsers = parser.add_subparsers(dest='command')
parser_start = subparsers.add_parser('start', help='to start a single node')
parser_start.add_argument("-n", "--node_id", default="node:0", help="the node id")
parser_up = subparsers.add_parser('up', help='to bring up all nodes')
parser_list = subparsers.add_parser('list', help='list node information')
parser_list.add_argument("-v", "--verbose", action='store_true', help="verbosely")

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        conf = json.load(file)

    nodes_def = conf["nodes"]
    devices_def = conf["devices"]

    if args.command == 'start':
        ppd.RPC.serve(args.node_id, nodes_def)
    elif args.command == 'up':
        workers = []
        for node_id in nodes_def.keys():
            worker = multiprocess.Process(
                target=ppd.RPC.serve, args=(node_id, nodes_def)
            )
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
    else:
        parser.print_help()
