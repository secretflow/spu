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


import subprocess
import time
import unittest
from socket import socket

import multiprocess

import spu.libspu.link as link
import spu.psi as psi
from spu.utils.simulation import PropagatingThread


def get_free_port():
    with socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def wc_count(file_name):
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])


class UnitTests(unittest.TestCase):
    def run_psi(self, fn):
        wsize = 2

        lctx_desc = link.Desc()
        for rank in range(wsize):
            lctx_desc.add_party(f"id_{rank}", f"thread_{rank}")

        def wrap(rank):
            lctx = link.create_mem(lctx_desc, rank)
            return fn(lctx)

        jobs = [PropagatingThread(target=wrap, args=(rank,)) for rank in range(wsize)]

        [job.start() for job in jobs]
        [job.join() for job in jobs]

    def run_streaming_psi(self, wsize, inputs, outputs, selected_fields, protocol):
        time_stamp = time.time()
        lctx_desc = link.Desc()
        lctx_desc.id = str(round(time_stamp * 1000))

        for rank in range(wsize):
            port = get_free_port()
            lctx_desc.add_party(f"id_{rank}", f"127.0.0.1:{port}")

        global wrap

        def wrap(rank, selected_fields, input_path, output_path, type):
            lctx = link.create_brpc(lctx_desc, rank)

            config = psi.BucketPsiConfig(
                psi_type=type,
                broadcast_result=True,
                input_params=psi.InputParams(
                    path=input_path, select_fields=selected_fields
                ),
                output_params=psi.OutputParams(path=output_path, need_sort=True),
                curve_type=psi.CurveType.CURVE_25519,
            )

            if type == psi.PsiType.DP_PSI_2PC:
                config.dppsi_params.bob_sub_sampling = 0.9
                config.dppsi_params.epsilon = 3

            report = psi.bucket_psi(lctx, config)

            source_count = wc_count(input_path)
            output_count = wc_count(output_path)
            print(
                f"id:{lctx.id()}, psi_type: {type}, original_count: {report.original_count}, intersection_count: {report.intersection_count}, source_count: {source_count}, output_count: {output_count}"
            )

            self.assertEqual(report.original_count, source_count - 1)
            self.assertEqual(report.intersection_count, output_count - 1)

            lctx.stop_link()

        # launch with multiprocess
        jobs = [
            multiprocess.Process(
                target=wrap,
                args=(
                    rank,
                    selected_fields,
                    inputs[rank],
                    outputs[rank],
                    protocol,
                ),
            )
            for rank in range(wsize)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)

    def prep_data(self):
        data = [
            [f'r{idx}' for idx in range(1000) if idx % 3 == 0],
            [f'r{idx}' for idx in range(1000) if idx % 7 == 0],
        ]

        expected = [f'r{idx}' for idx in range(1000) if idx % 3 == 0 and idx % 7 == 0]

        return data, expected

    def test_reveal(self):
        data, expected = self.prep_data()
        expected.sort()

        def fn(lctx):
            config = psi.MemoryPsiConfig(
                psi_type=psi.PsiType.ECDH_PSI_2PC, broadcast_result=True
            )
            joint = psi.mem_psi(lctx, config, data[lctx.rank])
            joint.sort()
            return self.assertEqual(joint, expected)

        self.run_psi(fn)

    def test_reveal_to(self):
        data, expected = self.prep_data()
        expected.sort()

        reveal_to_rank = 0

        def fn(lctx):
            config = psi.MemoryPsiConfig(
                psi_type=psi.PsiType.KKRT_PSI_2PC,
                receiver_rank=reveal_to_rank,
                broadcast_result=False,
            )
            joint = psi.mem_psi(lctx, config, data[lctx.rank])

            joint.sort()

            if lctx.rank == reveal_to_rank:
                self.assertEqual(joint, expected)
            else:
                self.assertEqual(joint, [])

        self.run_psi(fn)

    def test_ecdh_3pc(self):
        print("----------test_ecdh_3pc-------------")

        inputs = [
            "spu/tests/data/alice.csv",
            "spu/tests/data/bob.csv",
            "spu/tests/data/carol.csv",
        ]
        outputs = ["./alice-ecdh3pc.csv", "./bob-ecdh3pc.csv", "./carol-ecdh3pc.csv"]
        selected_fields = ["id", "idx"]

        self.run_streaming_psi(
            3, inputs, outputs, selected_fields, psi.PsiType.ECDH_PSI_3PC
        )

    def test_kkrt_2pc(self):
        print("----------test_kkrt_2pc-------------")

        inputs = ["spu/tests/data/alice.csv", "spu/tests/data/bob.csv"]
        outputs = ["./alice-kkrt.csv", "./bob-kkrt.csv"]
        selected_fields = ["id", "idx"]

        self.run_streaming_psi(
            2, inputs, outputs, selected_fields, psi.PsiType.KKRT_PSI_2PC
        )

    def test_ecdh_2pc(self):
        print("----------test_ecdh_2pc-------------")

        inputs = ["spu/tests/data/alice.csv", "spu/tests/data/bob.csv"]
        outputs = ["./alice-ecdh.csv", "./bob-ecdh.csv"]
        selected_fields = ["id", "idx"]

        self.run_streaming_psi(
            2, inputs, outputs, selected_fields, psi.PsiType.ECDH_PSI_2PC
        )

    def test_dppsi_2pc(self):
        print("----------test_dppsi_2pc-------------")

        inputs = ["spu/tests/data/alice.csv", "spu/tests/data/bob.csv"]
        outputs = ["./alice-dppsi.csv", "./bob-dppsi.csv"]
        selected_fields = ["id", "idx"]

        self.run_streaming_psi(
            2, inputs, outputs, selected_fields, psi.PsiType.DP_PSI_2PC
        )

    def test_ecdh_oprf_unbalanced(self):
        print("----------test_ecdh_oprf_unbalanced-------------")

        offline_path = ["", "spu/tests/data/bob.csv"]
        online_path = ["spu/tests/data/alice.csv", "spu/tests/data/bob.csv"]
        outputs = ["./alice-ecdh-unbalanced.csv", "./bob-ecdh-unbalanced.csv"]
        preprocess_path = ["./alice-preprocess.csv", ""]
        secret_key_path = ["", "./secret_key.bin"]
        selected_fields = ["id", "idx"]

        with open(secret_key_path[1], 'wb') as f:
            f.write(
                bytes.fromhex(
                    "000102030405060708090a0b0c0d0e0ff0e0d0c0b0a090807060504030201000"
                )
            )

        time_stamp = time.time()
        lctx_desc = link.Desc()
        lctx_desc.id = str(round(time_stamp * 1000))

        for rank in range(2):
            port = get_free_port()
            lctx_desc.add_party(f"id_{rank}", f"127.0.0.1:{port}")

        receiver_rank = 0
        server_rank = 1
        client_rank = 0
        # one-way PSI, just one party get result
        broadcast_result = False

        precheck_input = False
        server_cache_path = "server_cache.bin"

        global wrap

        def wrap(
            rank,
            offline_path,
            online_path,
            out_path,
            preprocess_path,
            ub_secret_key_path,
        ):
            link_ctx = link.create_brpc(lctx_desc, rank)

            if receiver_rank != link_ctx.rank:
                print("===== gen cache phase =====")
                print(f"{offline_path}, {server_cache_path}")

                gen_cache_config = psi.BucketPsiConfig(
                    psi_type=psi.PsiType.Value('ECDH_OPRF_UB_PSI_2PC_GEN_CACHE'),
                    input_params=psi.InputParams(
                        path=offline_path,
                        select_fields=selected_fields,
                        precheck=False,
                    ),
                    output_params=psi.OutputParams(
                        path=server_cache_path, need_sort=False
                    ),
                    bucket_size=1000000,
                    curve_type=psi.CurveType.CURVE_FOURQ,
                    ecdh_secret_key_path=ub_secret_key_path,
                )

                start = time.time()

                gen_cache_report = psi.gen_cache_for_2pc_ub_psi(gen_cache_config)

                server_source_count = wc_count(offline_path)
                self.assertEqual(
                    gen_cache_report.original_count, server_source_count - 1
                )

                print(f"offline cost time: {time.time() - start}")
                print(
                    f"offline: rank: {rank} original_count: {gen_cache_report.original_count}"
                )

            print("===== transfer cache phase =====")
            transfer_cache_config = psi.BucketPsiConfig(
                psi_type=psi.PsiType.Value('ECDH_OPRF_UB_PSI_2PC_TRANSFER_CACHE'),
                broadcast_result=broadcast_result,
                receiver_rank=receiver_rank,
                input_params=psi.InputParams(
                    path=offline_path,
                    select_fields=selected_fields,
                    precheck=precheck_input,
                ),
                bucket_size=1000000,
                curve_type=psi.CurveType.CURVE_FOURQ,
            )

            if receiver_rank == link_ctx.rank:
                transfer_cache_config.preprocess_path = preprocess_path
            else:
                transfer_cache_config.input_params.path = server_cache_path

            print(
                f"rank:{link_ctx.rank} file:{transfer_cache_config.input_params.path}"
            )

            start = time.time()
            transfer_cache_report = psi.bucket_psi(link_ctx, transfer_cache_config)

            if receiver_rank != link_ctx.rank:
                server_source_count = wc_count(offline_path)
                self.assertEqual(
                    transfer_cache_report.original_count, server_source_count - 1
                )

            print(f"transfer cache cost time: {time.time() - start}")
            print(
                f"transfer cache: rank: {rank} original_count: {transfer_cache_report.original_count}"
            )

            print("===== shuffle online phase =====")
            shuffle_online_config = psi.BucketPsiConfig(
                psi_type=psi.PsiType.Value('ECDH_OPRF_UB_PSI_2PC_SHUFFLE_ONLINE'),
                broadcast_result=False,
                receiver_rank=server_rank,
                input_params=psi.InputParams(
                    path=online_path,
                    select_fields=selected_fields,
                    precheck=precheck_input,
                ),
                output_params=psi.OutputParams(path=out_path, need_sort=False),
                bucket_size=10000000,
                curve_type=psi.CurveType.CURVE_FOURQ,
            )

            if client_rank == link_ctx.rank:
                shuffle_online_config.preprocess_path = preprocess_path
            else:
                shuffle_online_config.preprocess_path = server_cache_path
                shuffle_online_config.ecdh_secret_key_path = ub_secret_key_path

            print(
                f"rank:{link_ctx.rank} file:{shuffle_online_config.input_params.path}"
            )

            start = time.time()
            shuffle_online_report = psi.bucket_psi(link_ctx, shuffle_online_config)

            if server_rank == link_ctx.rank:
                server_source_count = wc_count(offline_path)
                self.assertEqual(
                    shuffle_online_report.original_count, server_source_count - 1
                )

            print(f"shuffle online cost time: {time.time() - start}")
            print(
                f"shuffle online: rank: {rank} original_count: {shuffle_online_report.original_count}"
            )
            print(
                f"shuffle online: rank: {rank} intersection: {shuffle_online_report.intersection_count}"
            )

            print("===== offline phase =====")
            offline_config = psi.BucketPsiConfig(
                psi_type=psi.PsiType.Value('ECDH_OPRF_UB_PSI_2PC_OFFLINE'),
                broadcast_result=broadcast_result,
                receiver_rank=client_rank,
                input_params=psi.InputParams(
                    path=offline_path,
                    select_fields=selected_fields,
                    precheck=precheck_input,
                ),
                output_params=psi.OutputParams(path="fake.out", need_sort=False),
                bucket_size=1000000,
                curve_type=psi.CurveType.CURVE_FOURQ,
            )

            if client_rank == link_ctx.rank:
                offline_config.preprocess_path = preprocess_path
                offline_config.input_params.path = "dummy.csv"
            else:
                offline_config.ecdh_secret_key_path = ub_secret_key_path

            start = time.time()
            offline_report = psi.bucket_psi(link_ctx, offline_config)

            if receiver_rank != link_ctx.rank:
                server_source_count = wc_count(offline_path)
                self.assertEqual(offline_report.original_count, server_source_count - 1)

            print(f"offline cost time: {time.time() - start}")
            print(
                f"offline: rank: {rank} original_count: {offline_report.original_count}"
            )
            print(
                f"offline: rank: {rank} intersection_count: {offline_report.intersection_count}"
            )

            print("===== online phase =====")
            online_config = psi.BucketPsiConfig(
                psi_type=psi.PsiType.Value('ECDH_OPRF_UB_PSI_2PC_ONLINE'),
                broadcast_result=broadcast_result,
                receiver_rank=client_rank,
                input_params=psi.InputParams(
                    path=online_path,
                    select_fields=selected_fields,
                    precheck=precheck_input,
                ),
                output_params=psi.OutputParams(path=out_path, need_sort=False),
                bucket_size=300000,
                curve_type=psi.CurveType.CURVE_FOURQ,
            )

            if receiver_rank == link_ctx.rank:
                online_config.preprocess_path = preprocess_path
            else:
                online_config.ecdh_secret_key_path = ub_secret_key_path
                online_config.input_params.path = "dummy.csv"

            start = time.time()
            report_online = psi.bucket_psi(link_ctx, online_config)

            if receiver_rank == link_ctx.rank:
                client_source_count = wc_count(online_path)
                self.assertEqual(report_online.original_count, client_source_count - 1)

            print(f"online cost time: {time.time() - start}")
            print(f"online: rank:{rank} original_count: {report_online.original_count}")
            print(f"intersection_count: {report_online.intersection_count}")

            link_ctx.stop_link()

        # launch with multiprocess
        jobs = [
            multiprocess.Process(
                target=wrap,
                args=(
                    rank,
                    offline_path[rank],
                    online_path[rank],
                    outputs[rank],
                    preprocess_path[rank],
                    secret_key_path[rank],
                ),
            )
            for rank in range(2)
        ]
        [job.start() for job in jobs]
        for job in jobs:
            job.join()
            self.assertEqual(job.exitcode, 0)


if __name__ == '__main__':
    unittest.main()
