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


import inspect
import json
import logging
import os
import sys
import unittest
from time import perf_counter

import numpy.testing as npt
import pandas as pd

import spu.utils.distributed as ppd
import spu.utils.distributed_impl as ppd_impl
from spu.utils.polyfill import Process

with open("examples/python/conf/3pc.json", 'r') as file:
    conf = json.load(file)

logger = logging.getLogger(ppd_impl.__name__)
logger.setLevel(level=logging.WARN)
_test_perf_table = pd.DataFrame({'name': [], 'duration': []})


def add_profile_data(name, duration):
    global _test_perf_table

    # Save result to table
    new_row = pd.DataFrame({'name': name, 'duration': duration}, index=[0])
    _test_perf_table = pd.concat([_test_perf_table, new_row], ignore_index=True)


def profile_test_point(foo, *args, **kwargs):
    # Get function name
    test_fun_name = inspect.getmodule(foo).__name__

    # Run with perf counter
    start = perf_counter()
    result = foo(*args, **kwargs)
    end = perf_counter()

    # Save result to table
    add_profile_data(test_fun_name, end - start)
    return result


def save_perf_report():
    buf = _test_perf_table.to_csv(index=False)
    p = os.path.expanduser(os.path.join('~', '.ml_test_perf.csv'))
    with open(p, '+w') as f:
        f.write(buf)


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workers = []
        for node_id in conf["nodes"].keys():
            worker = Process(target=ppd.RPC.serve, args=(node_id, conf["nodes"]))
            worker.start()
            cls.workers.append(worker)
        import time

        # wait for all process serving.
        time.sleep(2)

        rt_config = conf["devices"]["SPU"]["config"]["runtime_config"]
        rt_config["enable_pphlo_profile"] = False
        rt_config["enable_hal_profile"] = False
        ppd.init(conf["nodes"], conf["devices"])

    @classmethod
    def tearDownClass(cls):
        for worker in cls.workers:
            worker.kill()

    def test_flax_mlp(self):
        from examples.python.ml.flax_mlp import flax_mlp

        param = profile_test_point(flax_mlp.run_on_spu)

        score = flax_mlp.compute_score(ppd.get(param), 'spu')
        self.assertGreater(score, 0.95)

    def test_flax_vae(self):
        from examples.python.ml.flax_vae import flax_vae

        flax_vae.args.num_epochs = 1
        flax_vae.args.num_steps = 10
        metrics = profile_test_point(flax_vae.train)

        self.assertTrue(250 < metrics['loss'] < 280)
        self.assertTrue(235 < metrics['bce'] < 250)
        self.assertTrue(18 < metrics['kld'] < 30)

    def test_stax_mnist_classifier(self):
        from examples.python.ml.stax_mnist_classifier import stax_mnist_classifier

        stax_mnist_classifier.num_batches = 10
        accuracy = profile_test_point(stax_mnist_classifier.run_spu)
        self.assertGreater(accuracy, 0.7)

    def test_stax_nn(self):
        from examples.python.ml.stax_nn import stax_nn

        stax_nn.args.epoch = 1
        score = profile_test_point(
            stax_nn.run_model, model_name="network_a", run_cpu=False
        )
        self.assertGreater(score, 0.9)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(UnitTests('test_flax_mlp'))
    suite.addTest(UnitTests('test_flax_vae'))
    suite.addTest(UnitTests('test_stax_mnist_classifier'))
    suite.addTest(UnitTests('test_stax_nn'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(failfast=True)
    ret = not runner.run(suite()).wasSuccessful()

    save_perf_report()

    sys.exit(ret)
