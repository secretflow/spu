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

from __future__ import annotations

import concurrent.futures
import json
import logging
import sys
import traceback
import uuid
from functools import partial, wraps
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Sequence,
    Tuple,
    Type,
    Union,
)

import cloudpickle as pickle
import grpc
import jax
import numpy as np
from google.protobuf import json_format
from jax import linear_util as lu
from jax._src import api_util as japi_util
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

import spu.binding._lib.link as liblink
from spu.binding.api import Io, Runtime, compile
from spu.binding.util.distributed_pb2 import RunRequest, RunResponse
from spu.binding.util.distributed_pb2_grpc import (
    NodeServiceServicer,
    NodeServiceStub,
    add_NodeServiceServicer_to_server,
)
from spu.spu_pb2 import ExecutableProto, IrProto, RuntimeConfig, ValueProto

"""
This module is used as a simple scheduler to demonstrate SPU usage.
It's not designed for production both for performance and security reasons.

To use SPU in production, please consider SecretFlow instead.
"""


class ObjectRef:
    # Use a static random id to solve pickle+isinstance issue.
    KLASS_ID = '3fae14a7-66d6-48d6-b2c8-867a7b78af6e'

    def __init__(self, uuid: str, origin_nodeid: str):
        """
        uuid: the uuid in the whole distributed system.
        origin_nodeid: the origin node id which create this object.
        """
        self.uuid = uuid
        self.origin_nodeid = origin_nodeid

    def __repr__(self):
        return f"ObjRef({self.uuid} at {self.origin_nodeid})"

    def __hash__(self):
        return hash((self.uuid, self.origin_nodeid))

    def __eq__(self, other):
        return (self.uuid, self.origin_nodeid) == (other.uuid, other.origin_nodeid)


def isObjectRef(obj):
    return getattr(obj, 'KLASS_ID', None) == ObjectRef.KLASS_ID


class RPC:
    """A simple RPC wrapper"""

    OPTIONS = [
        ('grpc.max_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024),
    ]
    CHUNK_SIZE = 10 * 1024 * 1024

    @classmethod
    def makeStub(cls, addr):
        return NodeServiceStub(grpc.insecure_channel(addr, options=RPC.OPTIONS))

    @classmethod
    def serve(cls, node_id: str, nodes_def: Dict[str, str]):
        server = grpc.server(
            concurrent.futures.ThreadPoolExecutor(max_workers=10), options=RPC.OPTIONS
        )
        add_NodeServiceServicer_to_server(NodeServicer(node_id, nodes_def), server)
        logging.info(f"Starting grpc server at {nodes_def[node_id]}")
        server.add_insecure_port(nodes_def[node_id])
        server.start()
        server.wait_for_termination()


def split_message(msg: bytes) -> Iterable[bytes]:
    for offset in range(0, len(msg), RPC.CHUNK_SIZE):
        yield msg[offset : offset + RPC.CHUNK_SIZE]


def rebuild_messages(msgs: Generator[bytes, None, None]) -> bytes:
    return b''.join([msg for msg in msgs])


class NodeClient:
    def __init__(self, node_id: str, addr: str):
        self.node_id = node_id
        self.addr = addr
        self._stub = RPC.makeStub(addr)

    def _check_args(self, *args, **kwargs):
        def _check(x):
            assert not isinstance(x, Device.Object)

        tree_map(_check, (args, kwargs))

    def _call(self, stub_method, fn, *args, **kwargs):
        payload = pickle.dumps((fn, args, kwargs))
        rsp_gen = stub_method(
            RunRequest(data=split) for split in split_message(payload)
        )
        rsp_data = rebuild_messages(rsp_itr.data for rsp_itr in rsp_gen)
        result = pickle.loads(rsp_data)
        if isinstance(result, Exception):
            raise Exception("remote exception", result)
        else:
            return result

    def run(self, fn, *args, **kwargs):
        """Run a function on the corresponding node server"""
        self._check_args(*args, **kwargs)
        return self._call(self._stub.Run, fn, *args, **kwargs)

    def run_return(self, fn, *args, **kwargs):
        """Run a function on the corresponding node server"""
        self._check_args(*args, **kwargs)
        return self._call(self._stub.RunReturn, fn, *args, **kwargs)

    def get(self, ref: ObjectRef):
        def builtin_fetch_object(server, refid: str):
            return server._globals[ObjectRef(refid, server.node_id)]

        # use uuid directly to prevent server fetch object ref.
        return self._call(self._stub.RunReturn, builtin_fetch_object, ref.uuid)


class NodeServicer(NodeServiceServicer):
    def __init__(self, node_id: str, nodes_def: Dict[str, str]):
        self.node_id = node_id
        self.nodes_def = nodes_def

        # _locals saves objects visible only for this node.
        self._locals: Dict[str, object] = {}
        # _globals saves objects visible for the entire cluster.
        self._globals: Dict[ObjectRef, object] = {}
        # _node_clients used to communicate with other nodes in the cluster.
        self._node_clients = {
            node_id: NodeClient(node_id, addr) for node_id, addr in nodes_def.items()
        }

    def RunReturn(self, req_itr, ctx):
        payload = rebuild_messages(itr.data for itr in req_itr)
        (fn, args, kwargs) = pickle.loads(payload)
        logging.info(f"RunR: {fn.__name__} at {self.node_id}")
        try:
            args, kwargs = tree_map(lambda obj: self._get_object(obj), (args, kwargs))
            result = fn(self, *args, **kwargs)
            response = pickle.dumps(result)
        except Exception as e:
            stack_info = traceback.format_exc()
            logging.info(stack_info)
            response = pickle.dumps(Exception(stack_info))
        for split in split_message(response):
            yield RunResponse(data=split)

    def Run(self, req_itr, ctx):
        payload = rebuild_messages(itr.data for itr in req_itr)
        (fn, args, kwargs) = pickle.loads(payload)
        logging.info(f"Run : {fn.__name__} at {self.node_id}")
        try:
            args, kwargs = tree_map(lambda obj: self._get_object(obj), (args, kwargs))
            ret_objs = fn(self, *args, **kwargs)
            ret_refs = tree_map(lambda obj: self._add_object(obj), ret_objs)
            response = pickle.dumps(ret_refs)
        except Exception:
            stack_info = traceback.format_exc()
            logging.info(stack_info)
            response = pickle.dumps(Exception(stack_info))
        for split in split_message(response):
            yield RunResponse(data=split)

    def _get_object(self, ref: Union[ObjectRef, Any]):
        """Get an object from the distributed context."""
        if not isObjectRef(ref):
            return ref

        if ref in self._globals:
            # if the object is found in local database, return it.
            return self._globals[ref]
        else:
            obj = self._node_clients[ref.origin_nodeid].get(ref)
            self._globals[ref] = obj
            return obj

    def _add_object(self, obj: Any) -> ObjectRef:
        """Add an object to the cluster."""
        # we could also ignore it if it's already an object ref, we assert here for more strict semantic.
        assert not isObjectRef(obj)
        ref = ObjectRef(str(uuid.uuid4()), self.node_id)
        self._globals[ref] = obj
        return ref

    def _del_object(self, ref: ObjectRef) -> None:
        # the object may not on this node yet, just ignore.
        self._globals.pop(ref, None)


#####################################################################################
# The following parts define the virtual device module.
#
# All class and object are lived in `host` space.
#####################################################################################


def shape_spu_to_np(spu_shape):
    # x : spu_pb2.ShapeProto):
    return tuple(list(spu_shape.dims))


def dtype_spu_to_np(spu_dtype):
    from spu import spu_pb2

    MAP = {
        spu_pb2.DataType.DT_FXP: np.float32,
        spu_pb2.DataType.DT_I1: np.bool_,
        spu_pb2.DataType.DT_I8: np.int8,
        spu_pb2.DataType.DT_U8: np.uint8,
        spu_pb2.DataType.DT_I16: np.int16,
        spu_pb2.DataType.DT_U16: np.uint16,
        spu_pb2.DataType.DT_I32: np.int32,
        spu_pb2.DataType.DT_U32: np.uint32,
        spu_pb2.DataType.DT_I64: np.int64,
        spu_pb2.DataType.DT_U64: np.uint64,
    }
    return MAP.get(spu_dtype)


class Device:
    """A device is a virtual concept hosted by a list of nodes."""

    TRANS_KERNELS = {}

    class Object:
        """A device object is a handle lives on the host, which points to device resource"""

        def __init__(self, device: Device):
            self.device = device

        def __repr__(self):
            return f"DeviceObject({id(self)} at {self.device.name})"

    class Function:
        def __init__(self, device: Device, pyfunc: Callable):
            self.device = device
            self.pyfunc = pyfunc

        def __repr__(self):
            return f"DeviceFunction({id(self)} at {self.device.name})"

        def __call__(self, *args, **kwargs):
            pass

    def __init__(self, host: HostContext, name: str):
        self.host = host
        self.name = name

    def __call__(self, fn: Callable):
        """Device as a decorator, convert a pyfunc to a device function"""
        return self.compile(fn)

    def _place_arguments(self, *args, **kwargs):
        # place arguments onto this device.
        def place(obj):
            if not isinstance(obj, Device.Object):
                return obj
            return Device.move(obj, self)

        return tree_map(place, (args, kwargs))

    def _inc_objref(self, ref: ObjectRef):
        self.host._inc_objref(ref)

    def _dec_objref(self, ref: ObjectRef):
        self.host._dec_objref(ref)

    def get(self, obj: Device.Object):
        """Get this device object to the host"""

    def compile(self, fn: Callable, *comp_args, **comp_kwargs) -> Callable:
        """Compile a python callable to device callable"""

    @classmethod
    def move(cls, obj: Device.Object, dest: Device):
        """Move a device object to another device.

        The ObjectRef system can do lazy fetch, so we transfer ObjectRef only.
        """
        move_fn = Device.TRANS_KERNELS[obj.device.__class__, dest.__class__]
        return move_fn(dest, obj)


class PYU(Device):
    """ """

    class Object(Device.Object):
        device: PYU

        def __init__(self, device: PYU, ref: ObjectRef):
            super().__init__(device)
            self.ref = ref

            self.device._inc_objref(ref)

        def __del__(self):
            self.device._dec_objref(self.ref)

    class Function(Device.Function):
        device: PYU  # PEP-526

        def __init__(self, device: PYU, pyfunc: Callable):
            super().__init__(device, pyfunc)

        def __call__(self, *args, **kwargs):
            args, kwargs = self.device._place_arguments(*args, **kwargs)

            def prep_objref(x):
                if not isinstance(x, Device.Object):
                    return x
                if isinstance(x, PYU.Object):
                    return x.ref
                raise Exception(f"can not handle {x}")

            pyfunc = self.pyfunc

            @wraps(pyfunc)
            def server_fn(server, *args, **kwargs):
                return pyfunc(*args, **kwargs)

            args, kwargs = tree_map(prep_objref, (args, kwargs))

            return tree_map(
                partial(PYU.Object, self.device),
                self.device.node_client.run(server_fn, *args, **kwargs),
            )

    def __init__(self, host: HostContext, name: str, node_client: NodeClient):
        super().__init__(host, name)
        self.node_client = node_client

    def __repr__(self):
        return f"PYU({self.name}) hosted by: {self.node_client.addr}"

    def compile(self, fn: Callable):
        return PYU.Function(self, fn)

    def get(self, obj: PYU.Object):
        return self.node_client.get(obj.ref)


class ValueWrapper:
    """Workarounds for ValueProto could not be pickled."""

    def __init__(self, shape: Sequence[int], dtype: np.dtype, vtype, value_str: str):
        self.shape = shape
        self.dtype = dtype
        self.vtype = vtype
        self.value_str = value_str

    def __repr__(self):
        return f"ValueWrapper({self.shape},{self.dtype},{self.vtype})"


def builtin_spu_init(
    server, name: str, my_rank: int, addrs: List[str], spu_config_str: str
):
    if f"{name}-rt" in server._locals:
        logging.info(f"spu-runtime ({name}) already exist, reuse it")
        return
    desc = liblink.Desc()
    desc.recv_timeout_ms = 100 * 1000  # 100 seconds
    for rank, addr in enumerate(addrs):
        desc.add_party(f"r{rank}", addr)
    link = liblink.create_brpc(desc, my_rank)
    spu_config = RuntimeConfig()
    spu_config.ParseFromString(spu_config_str)
    if my_rank != 0:
        spu_config.enable_action_trace = False
        spu_config.enable_hal_profile = False
        spu_config.enable_pphlo_profile = False
    server._locals[f"{name}-rt"] = Runtime(link, spu_config)
    server._locals[f"{name}-io"] = Io(len(addrs), spu_config)
    logging.info(f"spu-runtime ({name}) initialized")


def builtin_fetch_meta(server, vals: List[ValueWrapper]):
    return [(v.shape, v.dtype, v.vtype) for v in vals]


def builtin_spu_run(
    server,
    device_name: str,
    fn_name: str,
    pphlo_str: str,
    args_flat: List[Union[ObjectRef, Any]],
    num_returns: int,
):
    from spu.spu_pb2 import Visibility

    rt = server._locals[f"{device_name}-rt"]
    io = server._locals[f"{device_name}-io"]

    spu_ir = IrProto()
    spu_ir.ParseFromString(pphlo_str)

    # mock input, output names.
    in_names = [f'{id(fn_name)}-in{idx}' for idx in range(len(args_flat))]
    out_names = [f'{id(fn_name)}-out{idx}' for idx in range(num_returns)]

    # make an spu executable.
    executable = ExecutableProto(
        name=fn_name,
        input_names=in_names,
        output_names=out_names,
        code=spu_ir.code,
    )

    # do infeed.
    for idx, arg in enumerate(args_flat):
        if isinstance(arg, ValueWrapper):
            v = ValueProto()
            v.ParseFromString(arg.value_str)
            rt.set_var(in_names[idx], v)
        else:
            arg = np.asarray(jax.numpy.asarray(arg))
            fst, *_ = io.make_shares(arg, Visibility.VIS_PUBLIC)
            rt.set_var(in_names[idx], fst)

    # run
    rt.run(executable)

    # do outfeed
    ret_protos = [rt.get_var(name) for name in out_names]
    rets = [
        ValueWrapper(
            shape_spu_to_np(proto.shape),
            dtype_spu_to_np(proto.data_type),
            proto.visibility,
            proto.SerializeToString(),
        )
        for proto in ret_protos
    ]

    # cleanup
    for name in in_names + out_names:
        rt.del_var(name)

    return rets


from spu import spu_pb2


class SPU(Device):
    class Object(Device.Object):
        device: SPU

        def __init__(
            self,
            device: SPU,
            refs: Sequence[ObjectRef],
            shape: Sequence[int],
            dtype: np.dtype,
            vtype: spu_pb2.Visibility,
        ):
            super().__init__(device)
            assert all(isObjectRef(ref) for ref in refs)
            # note: the refs could also be located on the node which does not host SPU.
            self.refs = refs

            for ref in refs:
                self.device._inc_objref(ref)

            self.dtype = dtype
            self.shape = shape
            self.vtype = vtype

        def __del__(self):
            for ref in self.refs:
                self.device._dec_objref(ref)

    class JaxFunction(Device.Function):
        device: SPU

        def __init__(self, device: SPU, pyfunc: Callable, static_argnums):
            super().__init__(device, pyfunc)
            self.static_argnums = static_argnums

        def __call__(self, *args, **kwargs):
            args, kwargs = self.device._place_arguments(*args, **kwargs)

            # now, all object are either PyObject or SPU.DeviceObject
            pphlo_ir, args_flat, out_shape = self._compile_jax_func(
                self.pyfunc, self.static_argnums, *args, **kwargs
            )
            _, out_tree = jax.tree_util.tree_flatten(out_shape)

            def get_share_ref(idx, obj):
                return obj.refs[idx] if isinstance(obj, SPU.Object) else obj

            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for idx, _ in enumerate(self.device.node_clients):
                    idx_args_flat = [get_share_ref(idx, arg) for arg in args_flat]
                    futures.append(
                        executor.submit(
                            self.device.node_clients[idx].run,
                            wraps(self.pyfunc)(builtin_spu_run),
                            self.device.name,
                            repr(self.pyfunc),
                            pphlo_ir.SerializeToString(),
                            idx_args_flat,
                            out_tree.num_leaves,
                        )
                    )
            results = [future.result() for future in futures]

            # fetch the result metas, since all nodes are symmetric, query node[0] is enough.
            metas = self.device.node_clients[0].run_return(
                builtin_fetch_meta, results[0]
            )

            ret_flat = [
                SPU.Object(self.device, share_refs, *meta)
                for share_refs, meta in zip(zip(*results), metas)
            ]
            return tree_unflatten(out_tree, ret_flat)

        def dump_pphlo(self, *args, **kwargs):
            args, kwargs = self.device._place_arguments(*args, **kwargs)
            pphlo_ir, *_ = self._compile_jax_func(
                self.pyfunc, self.static_argnums, *args, **kwargs
            )
            return pphlo_ir.code.decode('utf-8')

        def _compile_jax_func(self, fn, static_argnums, *args, **kwargs):
            def mock_parameters(obj: Union[SPU.Object, np.ndarray]):
                if isinstance(obj, SPU.Object):
                    return np.zeros(shape=obj.shape, dtype=obj.dtype)
                else:
                    assert not isinstance(obj, Device.Object)
                    return obj

            mock_args, mock_kwargs = tree_map(mock_parameters, (args, kwargs))
            xc, out_shape = jax.xla_computation(
                fn,
                backend="interpreter",
                return_shape=True,
                static_argnums=static_argnums,
            )(*mock_args, **mock_kwargs)

            ## code copied from jax.xla_computation to make args aligned.
            _, dyn_args = japi_util.argnums_partial_except(
                lu.wrap_init(fn), static_argnums, args, allow_invalid=False
            )
            args_flat, _ = jax.tree_util.tree_flatten((dyn_args, kwargs))
            # end copy from jax.xla_computation

            from spu.spu_pb2 import IrType, Visibility, XlaMeta

            # compile xla to pphlo
            xla_ir = IrProto(
                ir_type=IrType.IR_XLA_HLO,
                code=xc.as_serialized_hlo_module_proto(),
                meta=XlaMeta(
                    inputs=[
                        arg.vtype if isinstance(arg, SPU.Object) else spu_pb2.VIS_PUBLIC
                        for arg in args_flat
                    ]
                ),
            )
            pphlo_ir = compile(xla_ir)

            return pphlo_ir, args_flat, out_shape

    def __init__(
        self,
        host: HostContext,
        name: str,
        node_clients: List[NodeClient],
        internal_addrs: List[str],
        runtime_config: Dict,
    ):
        super().__init__(host, name)
        self.node_clients = node_clients

        self.internal_addrs = internal_addrs
        assert len(internal_addrs) == len(node_clients)

        self.runtime_config = json_format.Parse(
            json.dumps(runtime_config), RuntimeConfig()
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.node_clients[idx].run,
                    builtin_spu_init,
                    self.name,
                    idx,
                    internal_addrs,
                    self.runtime_config.SerializeToString(),
                )
                for idx, _ in enumerate(node_clients)
            ]
        results = [future.result() for future in futures]

    def __repr__(self):
        hosts = [nc.addr for nc in self.node_clients]
        return f"SPU({self.name}) hosted by: {hosts}"

    def details(self) -> str:
        hosts = [nc.addr for nc in self.node_clients]
        return f"name: {self.name}\nhosted by: {hosts}\ninternal addrs: {self.internal_addrs}\n{self.runtime_config}"

    def compile(self, fn: Callable, static_argnums=()) -> Callable:
        return SPU.JaxFunction(self, fn, static_argnums)

    def get(self, obj: SPU.Object):
        value_wrappers = [nc.get(ref) for nc, ref in zip(self.node_clients, obj.refs)]
        protos = []
        for wrapper in value_wrappers:
            proto = ValueProto()
            proto.ParseFromString(wrapper.value_str)
            protos.append(proto)
        io = Io(len(self.internal_addrs), self.runtime_config)
        return io.reconstruct(protos)


class HostContext:
    """A host controls multiple virtual devices."""

    def __init__(self, nodes_def: Dict[str, str], devices_def):
        self.nodes_def = nodes_def
        self.devices_def = devices_def
        self.node_clients = {
            node_id: NodeClient(node_id, addr) for node_id, addr in nodes_def.items()
        }
        self.devices = {}
        for name, detail in devices_def.items():
            if detail["kind"] == "PYU":
                self.devices[name] = PYU(
                    self, name, self.node_clients[detail["config"]["node_id"]]
                )
            elif detail["kind"] == "SPU":
                config = detail["config"]
                self.devices[name] = SPU(
                    self,
                    name,
                    [self.node_clients[node_id] for node_id in config["node_ids"]],
                    config["spu_internal_addrs"],
                    config["runtime_config"],
                )
            else:
                raise Exception("unknown kind {}".format(detail["kind"]))

        self._objrefs = Counter()
        self._dead_refs: List[ObjectRef] = []

    _GC_COLLECT_THRESHOLD = 50

    def _inc_objref(self, ref: ObjectRef):
        self._objrefs[ref] += 1

    def _dec_objref(self, ref: ObjectRef):
        self._objrefs[ref] -= 1

        # collect the dead_refs
        dead_refs = [k for k, v in self._objrefs.items() if v == 0]
        for ref in dead_refs:
            self._objrefs.pop(ref, None)
        self._dead_refs.extend(dead_refs)

        self._garbage_collect()

    def _garbage_collect(self):
        if len(self._dead_refs) < self._GC_COLLECT_THRESHOLD:
            return

        def builtin_gc(server, ref_pairs: List[Tuple[str, str]]):
            # pass Tuple[str, str] to prevent auto _get_object
            for uuid, from_nodeid in ref_pairs:
                server._del_object(ObjectRef(uuid, from_nodeid))

        try:
            # Note: `concurrent` maybe disposed before this call, but since the
            # program is about to exit we can ignore this kind of error.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        client.run_return,
                        builtin_gc,
                        [(ref.uuid, ref.origin_nodeid) for ref in self._dead_refs],
                    )
                    for _, client in self.node_clients.items()
                ]

            results = [future.result() for future in futures]
            self._dead_refs.clear()
        except:
            # Just ignore it, not good for production but enough for demonstration.
            pass


_CONTEXT: HostContext


def init(nodes_def, devices_def):
    """Init a multi-device layout on a given cluster."""
    global _CONTEXT
    _CONTEXT = HostContext(nodes_def, devices_def)


def current():
    """Get the current device context"""
    return _CONTEXT


def device(name):
    """Get the device handle."""
    return _CONTEXT.devices[name]


def get(args):
    """Get objects from device to this driver."""
    args_flat, tree = jax.tree_util.tree_flatten(args)
    out_flat = [
        arg.device.get(arg) if isinstance(arg, Device.Object) else arg
        for arg in args_flat
    ]
    return tree_unflatten(tree, out_flat)


from spu.spu_pb2 import Visibility


def PYU2PYU(to: PYU, obj: PYU.Object):
    return PYU.Object(to, obj.ref)


def SPU2PYU(to: PYU, obj: SPU.Object):
    # tell PYU the object refs, and run reconstruct on it.
    def reconstruct(wsize: int, spu_config_str: str, shares: List[ValueWrapper]):
        from spu.spu_pb2 import Visibility

        spu_config = RuntimeConfig()
        spu_config.ParseFromString(spu_config_str)
        spu_io = Io(wsize, spu_config)

        protos = []
        for share in shares:
            proto = ValueProto()
            proto.ParseFromString(share.value_str)
            protos.append(proto)

        return spu_io.reconstruct(protos)

    return to(reconstruct)(
        len(obj.device.node_clients),
        obj.device.runtime_config.SerializeToString(),
        [PYU.Object(to, ref) for ref in obj.refs],
    )


def PYU2SPU(to: SPU, obj: PYU.Object, vtype=Visibility.VIS_SECRET):
    # make shares on PYU, and tell SPU the object refs.
    def make_shares(wsize: int, spu_config_str: str, x: np.ndarray):
        spu_config = RuntimeConfig()
        spu_config.ParseFromString(spu_config_str)
        spu_io = Io(wsize, spu_config)

        # > x = np.array([1, 2])     # x.dtype = int64
        # > y = jnp.array([1, 2])    # y.dtype = int32
        # JAX(0.2.28) treats np.int64 as int32 at compilation time.
        # So we have to infeed int64 as in32 accordingly.
        x = np.asarray(jax.numpy.asarray(x))

        shares = spu_io.make_shares(x, vtype)
        return tuple(
            ValueWrapper(x.shape, x.dtype, vtype, share.SerializeToString())
            for share in shares
        )

    share_objs = obj.device(make_shares)(
        len(to.node_clients),
        to.runtime_config.SerializeToString(),
        obj,
    )
    metas = obj.device.node_client.run_return(builtin_fetch_meta, [share_objs[0].ref])
    return SPU.Object(to, [obj.ref for obj in share_objs], *metas[0])


def SPU2SPU(to: SPU, obj: SPU.Object):
    if obj.device != to:
        raise Exception("we only support 1 SPU for now!")

    return obj


Device.TRANS_KERNELS = {
    (PYU, PYU): PYU2PYU,
    (PYU, SPU): PYU2SPU,
    (SPU, PYU): SPU2PYU,  #  NOTE: AUTO REVEAL, for convenient.
    (SPU, SPU): SPU2SPU,
}

# Sample definitions

SAMPLE_NODES_DEF = {
    "node:0": "127.0.0.1:9327",
    "node:1": "127.0.0.1:9328",
    "node:2": "127.0.0.1:9329",
}


SAMPLE_DEVICES_DEF = {
    "SPU": {
        "kind": "SPU",
        "config": {
            "node_ids": ["node:0", "node:1", "node:2"],
            "spu_internal_addrs": [
                "127.0.0.1:9437",
                "127.0.0.1:9438",
                "127.0.0.1:9439",
            ],
            "runtime_config": {
                "protocol": "ABY3",
                "field": "FM128",
                "enable_pphlo_profile": True,
                # "enable_pphlo_trace": True,
            },
        },
    },
    "P1": {"kind": "PYU", "config": {"node_id": "node:0"}},
    "P2": {"kind": "PYU", "config": {"node_id": "node:1"}},
}

if __name__ == '__main__':
    import argparse

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

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s] [%(processName)s] %(message)s',
    )
    if args.command == 'start':
        RPC.serve(args.node_id, nodes_def)
    elif args.command == 'up':
        import multiprocessing

        workers = []
        for node_id in nodes_def.keys():
            worker = multiprocessing.Process(
                target=RPC.serve, args=(node_id, nodes_def)
            )
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()
    else:
        parser.print_help()
