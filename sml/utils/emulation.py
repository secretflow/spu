# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Emulation tries to emulate the function that run under certain protocol in a distributed environment.

import copy
import ipaddress
import json
import logging
import pathlib
import re
import subprocess
import time
from enum import Enum

import multiprocess
import yaml

# FIXME: remove this.
import examples.python.utils.dataset_utils as dsutil
import spu.utils.distributed as ppd

DATASET_MOCK_REGRESSION_BASIC = "examples/python/conf/ds_mock_regression_basic.json"
CLUSTER_ABY3_3PC = "examples/python/conf/3pc.json"
SML_HOME = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_CIDR = "172.16.238.0/24"
# FIXME: use a released image
SAMPLE_IMAGE = "secretflow/spu_emulator:0.0.1"
SAMPLE_DOCKER_COMPOSE_CONFIG = {
    "services": {},
    "networks": {
        "spu-emulation": {
            "name": "spu-emulation",
            "ipam": {
                "driver": "default",
                "config": [{"subnet": SAMPLE_CIDR, "gateway": "172.16.238.1"}],
            },
        }
    },
}
SAMPLE_DOCKER_NODE_CONFIG = {
    "image": SAMPLE_IMAGE,
    "ports": [],
    "volumes": [f"{SML_HOME.parent.resolve()}:/home/admin/dev/"],
    "command": "sh -c \"@0\"",
    "networks": {"spu-emulation": {"ipv4_address": None}},
    "cap_add": ["NET_ADMIN"],
}
SAMPLE_NETWORK_BANDWIDTH_CMD = (
    "tc qdisc add dev eth0 root handle 1: tbf rate @0mbit burst @0kb latency 800ms"
)
SAMPLE_NETWORK_LATENCY_CMD = (
    "tc qdisc add dev eth0 parent 1:1 handle 10: netem delay @0msec limit @1"
)
SAMPLE_NODE_LAUNCH_CMD = (
    "python3 /home/admin/dev/examples/python/utils/nodectl.py "
    "-c /home/admin/dev/sml/@0/emulation.json start --node_id @1 "
    "&> /home/admin/dev/sml/@0/@1.log"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
syslog = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(message)s')
syslog.setFormatter(formatter)
logger.addHandler(syslog)
logger.propagate = False


class Mode(Enum):
    MULTIPROCESS = 1
    DOCKER = 2


class Emulator:
    def __init__(
        self,
        cluster_config: str,
        mode: Mode = Mode.MULTIPROCESS,
        bandwidth: int = None,
        latency: int = None,
    ) -> None:
        assert mode in Mode, "Invalid emulator mode"
        self.mode = mode
        with open(cluster_config, 'r') as file:
            self.conf = json.load(file)
        self.bandwidth = bandwidth
        self.latency = latency

    def up(self):
        if self.mode == Mode.MULTIPROCESS:
            self._mode_multiprocess_up()
        else:
            self._mode_docker_up()
        time.sleep(1)
        ppd.init(self.conf["nodes"], self.conf["devices"])

    def down(self):
        if self.mode == Mode.MULTIPROCESS:
            self._mode_multiprocess_down()
        else:
            self._mode_docker_down()

    def _mode_multiprocess_up(self):
        logger.info("Start multiprocess cluster...")
        self.workers = []
        for node_id in self.conf["nodes"].keys():
            worker = multiprocess.Process(
                target=ppd.RPC.serve, args=(node_id, self.conf["nodes"])
            )
            worker.start()
            self.workers.append(worker)

    def _mode_multiprocess_down(self):
        logger.info("Shutdown multiprocess cluster...")
        for worker in self.workers:
            worker.terminate()

    @staticmethod
    def prepare_dataset(dataset_config_path: str):
        with open(dataset_config_path, "r") as f:
            dataset_config = json.load(f)

        # Mock: load dataset to this python runtime
        x1, x2, y = dsutil.load_dataset_by_config(dataset_config)

        # send (x1, y) from P1
        x1, y = ppd.device("P1")(dsutil.load_feature_r1)(x1, y)

        # send (x2, ) from P1
        x2 = ppd.device("P2")(dsutil.load_feature_r2)(x2)

        # return the references.
        return (x1, x2), y

    def _mode_docker_up(self):
        logger.info("Start docker cluster...")
        self._gen_config_file()

        self._run_cmd(
            [
                "docker-compose",
                "-f",
                self.emu_tmp_dir / "docker-compose.yml",
                "up",
                "-d",
            ]
        )

    def _mode_docker_down(self):
        logger.info("Shutdown docker cluster...")
        self._run_cmd(
            [
                "docker-compose",
                "-f",
                self.emu_tmp_dir / "docker-compose.yml",
                "down",
            ]
        )

    def run(self, func):
        def wrapper(*args, **kwargs):
            # run the func on SPU.
            res = ppd.device("SPU")(func)(*args, **kwargs)
            # reveal and return the result to caller.
            return ppd.get(res)

        return wrapper

    def _run_cmd(self, cmd):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
        )
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            logger.info(line.decode('utf-8'))
        exit_code = proc.wait()
        if exit_code != 0:
            raise Exception(f"Run cmd {cmd} failed")

    def _gen_config_file(self):
        self.network = ipaddress.ip_network(SAMPLE_CIDR)
        self.yaml = SAMPLE_DOCKER_COMPOSE_CONFIG
        self.emu_tmp_dir = (
            SML_HOME
            / f"emulation_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}"
        )
        self.emu_tmp_dir.mkdir(parents=True, exist_ok=True)

        # generate docker compose yaml
        for node_id, node_ip in zip(self.conf["nodes"].keys(), self.network.hosts()):
            # the first address has been used for docker subnet gateway
            node_ip += 1
            _, node_port = self.conf["nodes"][node_id].split(":")
            self.conf["nodes"][node_id] = f"{node_ip}:{node_port}"
            docker_node_yml = copy.deepcopy(SAMPLE_DOCKER_NODE_CONFIG)
            tc_cmd = ""
            if self.bandwidth:
                tc_cmd = f"{re.sub('@0', str(self.bandwidth), SAMPLE_NETWORK_BANDWIDTH_CMD)} && "
            if self.latency:
                tc_cmd += (
                    f"{re.sub('@0', str(self.latency), SAMPLE_NETWORK_LATENCY_CMD)} && "
                )
                # set bandwidth as 10Gbps if not set
                if not self.bandwidth:
                    self.bandwidth = 10 * 1000
                # ref: https://stackoverflow.com/questions/18792347/what-does-option-limit-in-tc-netem-mean-and-do
                tc_netem_limit = int(
                    self.bandwidth * 1000 * 1000 / 1280 / 8 * self.latency / 1000 * 2
                )
                tc_cmd = re.sub("@1", str(tc_netem_limit), tc_cmd)

            node_cmd = re.sub("@0", self.emu_tmp_dir.name, SAMPLE_NODE_LAUNCH_CMD)
            node_cmd = re.sub("@1", node_id, node_cmd)
            docker_node_yml["command"] = re.sub(
                "@0", tc_cmd + node_cmd, docker_node_yml["command"]
            )
            docker_node_yml["ports"].append(f"{node_port}:{node_port}")
            docker_node_yml["networks"]["spu-emulation"]["ipv4_address"] = str(node_ip)
            self.yaml["services"][re.sub(":", "_", node_id)] = docker_node_yml
        with open(self.emu_tmp_dir / "docker-compose.yml", 'w') as file:
            yaml.dump(self.yaml, file)

        # generate temporary SPU cluster config
        for i, (old_addr, new_ip) in enumerate(
            zip(
                self.conf["devices"]["SPU"]["config"]["spu_internal_addrs"],
                self.network.hosts(),
            )
        ):
            # the first address has been used for docker subnet gateway
            new_ip += 1
            self.conf["devices"]["SPU"]["config"]["spu_internal_addrs"][i] = re.sub(
                r"^[^:]*", str(new_ip), old_addr
            )
        with open(self.emu_tmp_dir / "emulation.json", "w") as outfile:
            json.dump(self.conf, outfile, indent=4)
