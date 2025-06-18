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

import unittest

import numpy as np
import numpy.testing as npt
from absl.testing import absltest, parameterized

import spu.api as ppapi
import spu.libspu as libspu


def _bytes_to_pb(msg: bytes):
    ret = libspu.ValueMeta()
    ret.ParseFromString(msg)
    return ret


@parameterized.product(
    wsize=(2, 3, 5),
    prot=(
        libspu.ProtocolKind.REF2K,
        libspu.ProtocolKind.SEMI2K,
        libspu.ProtocolKind.ABY3,
    ),
    field=(libspu.FieldType.FM64, libspu.FieldType.FM128),
    chunk_size=(4, 11, 33, 67, 127, 65535),
)
class UnitTests(parameterized.TestCase):
    def test_io(self, wsize, prot, field, chunk_size):
        if prot == libspu.ProtocolKind.ABY3 and wsize != 3:
            return

        config = libspu.RuntimeConfig(
            protocol=prot,
            field=field,
            fxp_fraction_bits=18,
        )
        config.share_max_chunk_size = chunk_size
        io = ppapi.Io(wsize, config)

        # SINT
        x = np.random.randint(10, size=(3, 4, 5))

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(
            _bytes_to_pb(xs[0].meta).shape.dims,
            [3, 4, 5],
        )
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)
        y = io.reconstruct(xs)

        npt.assert_equal(x, y)

        # SFXP
        x = np.random.rand(3, 4, 5)

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(
            _bytes_to_pb(xs[0].meta).shape.dims,
            [3, 4, 5],
        )
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)

        # PFXP
        x = np.random.rand(3, 4, 5)

        xs = io.make_shares(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(
            _bytes_to_pb(xs[0].meta).shape.dims,
            [3, 4, 5],
        )
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)

        # empty
        x = np.random.rand(1, 0)

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(
            _bytes_to_pb(xs[0].meta).shape.dims,
            [1, 0],
        )
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)

        self.assertEqual(len(xs[0].share_chunks), chunk_count)
        y = io.reconstruct(xs)

        npt.assert_almost_equal(x, y, decimal=5)

    def test_io_strides(self, wsize, prot, field, chunk_size):
        if prot == libspu.ProtocolKind.ABY3 and wsize != 3:
            return

        config = libspu.RuntimeConfig(protocol=prot, field=field, fxp_fraction_bits=18)
        config.share_max_chunk_size = chunk_size
        io = ppapi.Io(wsize, config)

        # SINT
        x = np.random.randint(10, size=(6, 7, 8))
        x = x[0:5:2, 0:7:2, 0:8:2]

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(
            _bytes_to_pb(xs[0].meta).shape.dims,
            [3, 4, 4],
        )
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)
        y = io.reconstruct(xs)

        npt.assert_equal(x, y)

        # SFXP
        x = np.random.rand(6, 7, 8)
        x = x[0:5:2, 0:7:2, 0:8:2]

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(
            _bytes_to_pb(xs[0].meta).shape.dims,
            [3, 4, 4],
        )
        y = io.reconstruct(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)

        npt.assert_almost_equal(x, y, decimal=5)

        # PFXP
        x = np.random.rand(6, 7, 8)
        x = x[0:5:2, 0:7:2, 0:8:2]

        xs = io.make_shares(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(
            _bytes_to_pb(xs[0].meta).shape.dims,
            [3, 4, 4],
        )
        y = io.reconstruct(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)

        npt.assert_almost_equal(x, y, decimal=5)

    def test_io_scalar(self, wsize, prot, field, chunk_size):
        if prot == libspu.ProtocolKind.ABY3 and wsize != 3:
            return

        config = libspu.RuntimeConfig(protocol=prot, field=field, fxp_fraction_bits=18)
        config.share_max_chunk_size = chunk_size
        io = ppapi.Io(wsize, config)

        # SINT
        x = np.random.randint(10, size=())

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(_bytes_to_pb(xs[0].meta).shape.dims, [])
        y = io.reconstruct(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)

        npt.assert_equal(x, y)

        # SFXP
        x = np.random.random(size=())

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(_bytes_to_pb(xs[0].meta).shape.dims, [])
        y = io.reconstruct(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)

        npt.assert_almost_equal(x, y, decimal=5)

        # PFXP
        x = np.random.random(size=())

        xs = io.make_shares(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs), wsize)
        self.assertEqual(_bytes_to_pb(xs[0].meta).shape.dims, [])
        y = io.reconstruct(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs[0].share_chunks), chunk_count)

        npt.assert_almost_equal(x, y, decimal=5)

    def test_io_single_complex(self, wsize, prot, field, chunk_size):
        if prot == libspu.ProtocolKind.ABY3 and wsize != 3:
            return

        config = libspu.RuntimeConfig(protocol=prot, field=field)
        config.share_max_chunk_size = chunk_size
        io = ppapi.Io(wsize, config)

        # SFXP
        x = np.array([1 + 2j, 3 + 4j, 5 + 6j]).astype('complex64')

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)

        y = io.reconstruct(xs)
        print(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs[0].share_chunks), 2 * chunk_count)

        npt.assert_almost_equal(x, y, decimal=5)

        # PFXP
        xs = io.make_shares(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs), wsize)
        y = io.reconstruct(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs[0].share_chunks), 2 * chunk_count)

        npt.assert_almost_equal(x, y, decimal=5)

    def test_io_double_complex(self, wsize, prot, field, chunk_size):
        if prot == libspu.ProtocolKind.ABY3 and wsize != 3:
            return

        config = libspu.RuntimeConfig(protocol=prot, field=field)
        config.share_max_chunk_size = chunk_size
        io = ppapi.Io(wsize, config)

        # SFXP
        x = np.array([1 + 2j, 3 + 4j, 5 + 6j])

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs), wsize)

        y = io.reconstruct(xs)
        print(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_SECRET)
        self.assertEqual(len(xs[0].share_chunks), 2 * chunk_count)

        npt.assert_almost_equal(x, y, decimal=5)

        # PFXP
        xs = io.make_shares(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs), wsize)
        y = io.reconstruct(xs)
        chunk_count = io.get_share_chunk_count(x, libspu.Visibility.VIS_PUBLIC)
        self.assertEqual(len(xs[0].share_chunks), 2 * chunk_count)

        npt.assert_almost_equal(x, y, decimal=5)

    def test_colocated_io(self, wsize, prot, field, chunk_size):
        if prot == libspu.ProtocolKind.ABY3 and wsize != 3:
            return

        if prot == libspu.ProtocolKind.REF2K:
            return

        config = libspu.RuntimeConfig(
            protocol=prot,
            field=field,
        )
        config.share_max_chunk_size = chunk_size
        config.experimental_enable_colocated_optimization = True
        io = ppapi.Io(wsize, config)

        # PrivINT
        x = np.random.randint(10, size=())

        xs = io.make_shares(x, libspu.Visibility.VIS_SECRET, owner_rank=1)
        self.assertIn('Priv2k', _bytes_to_pb(xs[0].meta).storage_type)
        y = io.reconstruct(xs)

        npt.assert_equal(x, y)


if __name__ == '__main__':
    unittest.main()
