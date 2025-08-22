# Copyright 2024 Ant Group Co., Ltd.
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

import subprocess
import time

from numpy.random import randint


def get_free_port():
    return randint(low=49152, high=65536)


def wc_count(file_name):
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])


def create_link_desc_dict(world_size: int) -> dict:
    time_stamp = time.time()
    link_id = str(round(time_stamp * 1000))
    link_ports = []
    for _ in range(world_size):
        link_ports.append(get_free_port())

    return {"id": link_id, "wsize": world_size, "ports": link_ports}


def build_link_desc(link_desc: dict):
    import spu.libspu.link as link

    lctx_desc = link.Desc()
    lctx_desc.id = link_desc["id"]
    link_ports = link_desc["ports"]
    for idx in range(link_desc["wsize"]):
        port = link_ports[idx]
        lctx_desc.add_party(f"id_{idx}", f"127.0.0.1:{port}")

    return lctx_desc
