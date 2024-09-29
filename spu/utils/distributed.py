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

from .distributed_impl import (  # type: ignore
    RPC,
    PYU,
    SPU,
    Framework,
    init,
    device,
    get,
    current,
    set_framework,
    SAMPLE_NODES_DEF,
    SAMPLE_DEVICES_DEF,
    dtype_spu_to_np,
    shape_spu_to_np,
    save,
    load,
)


def main():
    import argparse
    import json
    from spu.utils.polyfill import Process

    parser = argparse.ArgumentParser(description='SPU node service.')
    parser.add_argument("-c", "--config", default="", help="the config")
    subparsers = parser.add_subparsers(dest='command')
    parser_start = subparsers.add_parser('start', help='to start a single node')
    parser_start.add_argument("-n", "--node_id", default="node:0", help="the node id")
    parser_up = subparsers.add_parser('up', help='to bring up all nodes')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as file:
            conf = json.load(file)
        nodes_def = conf["nodes"]
        devices_def = conf["devices"]
    else:
        nodes_def = SAMPLE_NODES_DEF
        devices_def = SAMPLE_DEVICES_DEF

    if args.command == 'start':
        RPC.serve(args.node_id, nodes_def)
    elif args.command == 'up':
        workers = []
        for node_id in nodes_def.keys():
            worker = Process(target=RPC.serve, args=(node_id, nodes_def))
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
