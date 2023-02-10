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


import socket
import sys
import unittest
from contextlib import closing
from typing import cast

import jax.numpy as jnp
import multiprocess
import numpy as np
import numpy.testing as npt
import tensorflow as tf

import spu.utils.distributed as ppd
from spu import spu_pb2


def unused_tcp_port() -> int:
    """Return an unused port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("localhost", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return cast(int, sock.getsockname()[1])


TEST_NODES_DEF = {
    "node:0": f"127.0.0.1:{unused_tcp_port()}",
    "node:1": f"127.0.0.1:{unused_tcp_port()}",
    "node:2": f"127.0.0.1:{unused_tcp_port()}",
}


TEST_DEVICES_DEF = {
    "SPU": {
        "kind": "SPU",
        "config": {
            "node_ids": ["node:0", "node:1", "node:2"],
            "spu_internal_addrs": [
                f"127.0.0.1:{unused_tcp_port()}",
                f"127.0.0.1:{unused_tcp_port()}",
                f"127.0.0.1:{unused_tcp_port()}",
            ],
            "runtime_config": {
                "protocol": "ABY3",
                "field": "FM64",
                # "enable_pphlo_profile": True,
            },
        },
    },
    "P1": {"kind": "PYU", "config": {"node_id": "node:0"}},
    "P2": {"kind": "PYU", "config": {"node_id": "node:1"}},
    "P3": {"kind": "PYU", "config": {"node_id": "node:2"}},
}


def no_in_no_out():
    pass


def no_in_one_out():
    return np.array([1, 2])


def no_in_two_out():
    return np.array([1, 2]), np.array([3.0, 4.0])


def no_in_list_out():
    return [np.array([1, 2]), np.array([3.0, 4.0])]


def no_in_dict_out():
    return {"first": np.array([1, 2]), "second": np.array([3.0, 4.0])}


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workers = []
        for node_id in TEST_NODES_DEF.keys():
            worker = multiprocess.Process(
                target=ppd.RPC.serve, args=(node_id, TEST_NODES_DEF)
            )
            worker.start()
            cls.workers.append(worker)
        import time

        # wait for all process serving.
        time.sleep(0.05)

        ppd.init(TEST_NODES_DEF, TEST_DEVICES_DEF)

    @classmethod
    def tearDownClass(cls):
        for worker in cls.workers:
            worker.kill()

    def test_basic_pyu(self):
        # no in, one out
        a = ppd.device("P1")(no_in_one_out)()
        self.assertTrue(isinstance(a, ppd.PYU.Object))
        npt.assert_equal(ppd.get(a), np.array([1, 2]))

        # no in, two out
        a, b = ppd.device("P1")(no_in_two_out)()
        self.assertTrue(isinstance(a, ppd.PYU.Object))
        self.assertTrue(a.device is ppd.current().devices["P1"])
        self.assertTrue(isinstance(b, ppd.PYU.Object))
        self.assertTrue(b.device is ppd.current().devices["P1"])
        npt.assert_equal(ppd.get(a), np.array([1, 2]))
        npt.assert_equal(ppd.get(b), np.array([3.0, 4.0]))

        # no in, list out
        l = ppd.device("P1")(no_in_list_out)()
        self.assertEqual(len(l), 2)
        self.assertTrue(isinstance(l[0], ppd.PYU.Object))
        self.assertTrue(isinstance(l[1], ppd.PYU.Object))
        npt.assert_equal(ppd.get(l[0]), np.array([1, 2]))
        npt.assert_equal(ppd.get(l[1]), np.array([3.0, 4.0]))

        # no in, dict out
        d = ppd.device("P1")(no_in_dict_out)()
        self.assertTrue(isinstance(d["first"], ppd.PYU.Object))
        self.assertTrue(isinstance(d["second"], ppd.PYU.Object))
        npt.assert_equal(ppd.get(d["first"]), np.array([1, 2]))
        npt.assert_equal(ppd.get(d["second"]), np.array([3.0, 4.0]))

        # immediate input from driver
        e = ppd.device("P1")(jnp.add)(np.array([1, 2]), np.array([3, 4]))
        self.assertTrue(isinstance(e, ppd.PYU.Object))
        npt.assert_equal(ppd.get(e), np.array([4, 6]))

        # reuse inputs from same device
        c = ppd.device("P1")(np.add)(a, b)
        self.assertTrue(c.device is ppd.current().devices["P1"])
        npt.assert_equal(ppd.get(c), np.array([4.0, 6.0]))

        # run on P2, fetch inputs from P1
        x = ppd.device("P2")(np.add)(a, a)
        self.assertTrue(x.device is ppd.current().devices["P2"])
        npt.assert_equal(ppd.get(x), np.array([2, 4]))

        # run on P3, fetch from P1 & P2
        u = ppd.device("P3")(np.add)(a, x)
        self.assertTrue(u.device is ppd.current().devices["P3"])
        npt.assert_equal(ppd.get(u), np.array([3, 6]))

    def test_basic_spu_jax(self):
        a = ppd.device("SPU")(no_in_one_out)()
        self.assertTrue(isinstance(a, ppd.SPU.Object))
        self.assertEqual(a.vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(a), np.array([1, 2]))

        # no in, two out
        a, b = ppd.device("SPU")(no_in_two_out)()
        self.assertTrue(isinstance(a, ppd.SPU.Object))
        self.assertTrue(isinstance(b, ppd.SPU.Object))
        self.assertEqual(a.vtype, spu_pb2.VIS_PUBLIC)
        self.assertEqual(b.vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(a), np.array([1, 2]))
        npt.assert_equal(ppd.get(b), np.array([3.0, 4.0]))

        # no in, list out
        l = ppd.device("SPU")(no_in_list_out)()
        self.assertEqual(len(l), 2)
        self.assertTrue(isinstance(l[0], ppd.SPU.Object))
        self.assertTrue(isinstance(l[1], ppd.SPU.Object))
        self.assertEqual(l[0].vtype, spu_pb2.VIS_PUBLIC)
        self.assertEqual(l[1].vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(l[0]), np.array([1, 2]))
        npt.assert_equal(ppd.get(l[1]), np.array([3.0, 4.0]))

        # no in, dict out
        d = ppd.device("SPU")(no_in_dict_out)()
        self.assertTrue(isinstance(d["first"], ppd.SPU.Object))
        self.assertTrue(isinstance(d["second"], ppd.SPU.Object))
        self.assertEqual(d["first"].vtype, spu_pb2.VIS_PUBLIC)
        self.assertEqual(d["second"].vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(d["first"]), np.array([1, 2]))
        npt.assert_equal(ppd.get(d["second"]), np.array([3.0, 4.0]))

        # immediate input from driver
        e = ppd.device("SPU")(jnp.add)(np.array([1, 2]), np.array([3, 4]))
        self.assertTrue(isinstance(e, ppd.SPU.Object))
        self.assertEqual(e.vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(e), np.array([4, 6]))

        # reuse inputs from SPU
        c = ppd.device("SPU")(jnp.add)(a, b)
        self.assertTrue(isinstance(c, ppd.SPU.Object))
        self.assertEqual(c.vtype, spu_pb2.VIS_PUBLIC)
        self.assertTrue(c.device is ppd.current().devices["SPU"])
        npt.assert_equal(ppd.get(c), np.array([4.0, 6.0]))

        # reuse a from SPU, x from pyu
        x = ppd.device("P1")(no_in_one_out)()
        c = ppd.device("SPU")(jnp.add)(a, x)
        self.assertTrue(isinstance(c, ppd.SPU.Object))
        self.assertEqual(c.vtype, spu_pb2.VIS_SECRET)
        self.assertTrue(c.device is ppd.current().devices["SPU"])
        npt.assert_equal(ppd.get(c), np.array([2, 4]))

        # transfer c from SPU to PYU.
        y = ppd.device("P2")(jnp.add)(c, 1)
        self.assertTrue(y.device is ppd.current().devices["P2"])
        npt.assert_equal(ppd.get(y), np.array([3, 5]))

    def test_dump_pphlo(self):
        a, b = ppd.device("P1")(no_in_two_out)()
        x, y = ppd.device("SPU")(no_in_two_out)()

        # dump pphlo
        text = ppd.device("SPU")(jnp.add).dump_pphlo(a, x)
        self.assertIn('pphlo.add', text)

    def test_basic_spu_tf(self):
        ppd._FRAMEWORK = ppd.Framework.EXP_TF
        a = ppd.device("SPU")(no_in_one_out)()
        self.assertTrue(isinstance(a, ppd.SPU.Object))
        self.assertEqual(a.vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(a), np.array([1, 2]))

        # no in, two out
        a, b = ppd.device("SPU")(no_in_two_out)()
        self.assertTrue(isinstance(a, ppd.SPU.Object))
        self.assertTrue(isinstance(b, ppd.SPU.Object))
        self.assertEqual(a.vtype, spu_pb2.VIS_PUBLIC)
        self.assertEqual(b.vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(a), np.array([1, 2]))
        npt.assert_equal(ppd.get(b), np.array([3.0, 4.0]))

        # no in, list out
        l = ppd.device("SPU")(no_in_list_out)()
        self.assertEqual(len(l), 2)
        self.assertTrue(isinstance(l[0], ppd.SPU.Object))
        self.assertTrue(isinstance(l[1], ppd.SPU.Object))
        self.assertEqual(l[0].vtype, spu_pb2.VIS_PUBLIC)
        self.assertEqual(l[1].vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(l[0]), np.array([1, 2]))
        npt.assert_equal(ppd.get(l[1]), np.array([3.0, 4.0]))

        # no in, dict out
        d = ppd.device("SPU")(no_in_dict_out)()
        self.assertTrue(isinstance(d["first"], ppd.SPU.Object))
        self.assertTrue(isinstance(d["second"], ppd.SPU.Object))
        self.assertEqual(d["first"].vtype, spu_pb2.VIS_PUBLIC)
        self.assertEqual(d["second"].vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(d["first"]), np.array([1, 2]))
        npt.assert_equal(ppd.get(d["second"]), np.array([3.0, 4.0]))

        # immediate input from driver
        e = ppd.device("SPU")(tf.add)(np.array([1, 2]), np.array([3, 4]))
        self.assertTrue(isinstance(e, ppd.SPU.Object))
        self.assertEqual(e.vtype, spu_pb2.VIS_PUBLIC)
        npt.assert_equal(ppd.get(e), np.array([4, 6]))

        # reuse inputs from SPU
        c = ppd.device("SPU")(tf.add)(a, a)
        self.assertTrue(isinstance(c, ppd.SPU.Object))
        self.assertEqual(c.vtype, spu_pb2.VIS_PUBLIC)
        self.assertTrue(c.device is ppd.current().devices["SPU"])
        npt.assert_equal(ppd.get(c), np.array([2, 4]))

        # reuse a from SPU, x from pyu
        x = ppd.device("P1")(no_in_one_out)()
        c = ppd.device("SPU")(tf.add)(a, x)
        self.assertTrue(isinstance(c, ppd.SPU.Object))
        self.assertEqual(c.vtype, spu_pb2.VIS_SECRET)
        self.assertTrue(c.device is ppd.current().devices["SPU"])
        npt.assert_equal(ppd.get(c), np.array([2, 4]))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(UnitTests('test_basic_pyu'))
    suite.addTest(UnitTests('test_basic_spu_jax'))
    suite.addTest(UnitTests('test_dump_pphlo'))
    suite.addTest(UnitTests('test_basic_spu_tf'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    ret = not runner.run(suite()).wasSuccessful()
    sys.exit(ret)
