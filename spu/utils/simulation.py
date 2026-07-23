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

import os
# os.environ["XLA_FLAGS"] = "--xla_dump_hlo_pass_re=."
# os.environ["XLA_FLAGS"] = "--xla_dump_to=/home/leiyu.lyc/ppu/xla_dumps_svm_trace3"

import threading
from typing import Callable

import jax

try:
    import jax.extend.linear_util as jax_lu
except ImportError:
    import jax.linear_util as jax_lu  # fallback

import jax.numpy as jnp
import numpy as np
from jax._src import api_util as japi_util
import json
# import os

from .. import api as spu_api
from .. import libspu  # type: ignore
from .. import spu_pb2
from . import frontend as spu_fe

import re
from mlir.dialects import stablehlo
from mlir import ir

import time

from jax._src.lib import xla_extension as xla

# https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread-in-python
class PropagatingThread(threading.Thread):
    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.ret


class Simulator(object):
    def __init__(self, wsize: int, rt_config: spu_pb2.RuntimeConfig):
        self.wsize = wsize
        self.rt_config = rt_config
        self.io = spu_api.Io(wsize, rt_config)

    @classmethod
    def simple(cls, wsize: int, prot: spu_pb2.ProtocolKind, field: spu_pb2.FieldType):
        """helper method to create an SPU Simulator

        Args:
            wsize (int): the world size.

            prot (spu_pb2.ProtocolKind): protocol.

            field (spu_pb2.FieldType): field type.

        Returns:
            A SPU Simulator
        """
        config = spu_pb2.RuntimeConfig(protocol=prot, field=field)

        if prot == spu_pb2.ProtocolKind.CHEETAH:
            # config.cheetah_2pc_config.enable_mul_lsb_error = True
            # config.cheetah_2pc_config.ot_kind = spu_pb2.CheetahOtKind.YACL_Softspoken
            pass
        config.enable_hal_profile = True
        config.enable_pphlo_profile = True
        # config.enable_pphlo_trace = True
        # config.enable_action_trace = True
        # config.enable_type_checker = True
        return cls(wsize, config)

    def __call__(self, executable, *flat_args):
        flat_args = [np.array(jnp.array(x)) for x in flat_args]
        params = [
            self.io.make_shares(x, spu_pb2.Visibility.VIS_SECRET) for x in flat_args
        ]

        lctx_desc = libspu.link.Desc()
        for rank in range(self.wsize):
            lctx_desc.add_party(f"id_{rank}", f"thread_{rank}")

        def wrapper(rank):
            lctx = libspu.link.create_mem(lctx_desc, rank)
            rank_config = spu_pb2.RuntimeConfig()
            rank_config.CopyFrom(self.rt_config)
            # if rank != 0:
            #     rank_config.enable_pphlo_trace = False
            #     rank_config.enable_action_trace = False
            #     rank_config.enable_hal_profile = False
            #     rank_config.enable_pphlo_profile = False
            rank_config.enable_hal_profile = True
            rank_config.enable_pphlo_profile = True
            rt = spu_api.Runtime(lctx, rank_config)

            # do infeed.
            for idx, param in enumerate(params):
                rt.set_var(executable.input_names[idx], param[rank])

            # run
            rt.run(executable)

            # do outfeed
            return [rt.get_var(name) for name in executable.output_names]

        jobs = [
            PropagatingThread(target=wrapper, args=(rank,))
            for rank in range(self.wsize)
        ]

        [job.start() for job in jobs]
        parties = [job.join() for job in jobs]

        outputs = zip(*parties)
        return [self.io.reconstruct(out) for out in outputs]

def tensor_string_to_ndarray(tensor_str):
    """
    Convert a tensor string to a NumPy ndarray.

    Parameters:
    - tensor_str (str): The tensor string to convert.

    Returns:
    - np.ndarray: The corresponding NumPy array.
    """
    # Step 1: Extract the metadata (shape and dtype) from the tensor declaration
    metadata_pattern = r'tensor<([\dxafa-f0-9]+)>'
    metadata_match = re.search(metadata_pattern, tensor_str)
    if not metadata_match:
        raise ValueError("Invalid tensor format: Missing tensor metadata.")
    
    metadata = metadata_match.group(1)
    
    # Split the metadata by 'x' to get dimensions and dtype
    metadata_parts = metadata.split('x')
    if len(metadata_parts) < 2:
        raise ValueError("Invalid tensor metadata: Not enough dimensions specified.")
    
    # The last part is the dtype
    dtype_str = metadata_parts[-1]
    shape = tuple(int(dim) for dim in metadata_parts[:-1])
    
    # Map the dtype string to NumPy dtype
    dtype_mapping = {
        'f32': np.float32,
        'f64': np.float64,
        'i32': np.int32,
        'i64': np.int64,
        'u8':  np.uint8,
        # Add more mappings as needed
    }
    
    if dtype_str not in dtype_mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    dtype = dtype_mapping[dtype_str]
    
    # Step 2: Extract the data block
    data_pattern = r'\{([\s\S]+)\}'
    data_match = re.search(data_pattern, tensor_str)
    if not data_match:
        raise ValueError("Invalid tensor format: Missing data block.")
    
    data_str = data_match.group(1).strip()
    
    # Replace curly brackets with square brackets for JSON compatibility
    data_str = data_str.replace('{', '[').replace('}', ']')
    
    try:
        # Safely evaluate the list using eval with restricted globals
        data = eval(data_str, {"__builtins__": None}, {})
    except Exception as e:
        raise ValueError(f"Error parsing data block: {e}")
    
    # Step 3: Convert the data to a NumPy array
    array = np.array(data, dtype=dtype)
    
    # Step 4: Validate the shape
    if array.shape != shape:
        raise ValueError(f"Shape mismatch: Expected {shape}, got {array.shape}")
    
    return array

def sim_jax(
    sim: Simulator,
    fun: Callable,
    static_argnums=(),
    copts=spu_pb2.CompilerOptions(),
    pphlo_ref = (None, None),
    skip = False,
    skip_list = None,
    not_skip_list = None,
    backend = "pphlo",
    time_record = False,
    hlo_log = False
):
    """
    Decorates a jax numpy fn that simulated on SPU.

        >>> sim = Simulator.simple(3, spu_pb2.ProtocolKind.ABY3, spu_pb2.FieldType.FM64)
        >>> spu_fn = sim_jax(sim, jnp.add)

    Then we can call spu_fn like normal jnp fn.

        >>> x = np.array([[1, 2], [3, 4]])
        >>> y = np.array([[5, 6], [7, 8]])
        >>> z = spu_fn(x, y)

    The function will be evaluated in an spu simulator.
    """

    def wrapper(*args, **kwargs):
        if skip:
            print(f"{fun.__name__} is skipped")
            return "skipped"
        if skip_list and fun.__name__ in skip_list:
            print(f"{fun.__name__} is skipped")
            return "skipped"
        if not_skip_list and fun.__name__ not in not_skip_list:
            print(f"{fun.__name__} is skipped")
            return "skipped"
        if backend == "cpu" or backend == "interpreter":
            copts.testplaintext = True
        else:
            # When the same sim is used for multiple functions, the testplaintext should be reset to False
            copts.testplaintext = False
        _, dyn_args = japi_util.argnums_partial_except(
            jax_lu.wrap_init(fun), static_argnums, args, allow_invalid=False
        )
        args_flat, _ = jax.tree_util.tree_flatten((dyn_args, kwargs))

        in_names = [f'in{idx}' for idx in range(len(args_flat))]

        def outputNameGen(out_flat):
            return [f'out{idx}' for idx in range(len(out_flat))]

        executable, output = spu_fe.compile(
            spu_fe.Kind.JAX,
            fun,
            args,
            kwargs,
            in_names,
            [spu_pb2.Visibility.VIS_SECRET] * len(args_flat),
            outputNameGen,
            static_argnums=static_argnums,
            copts=copts,
            hlo_log=hlo_log
        )
        
        pphlo_cur = executable.code.decode("utf-8")
        if backend == "cpu":
            cpu_backend = jax.lib.xla_bridge.get_backend("cpu")
            cpu_executable = cpu_backend.compile(pphlo_cur)
            def to_buffer(arr):
                if isinstance(arr, np.ndarray):
                    dtype = str(arr.dtype)
                    if dtype.startswith("float"):
                        arg = np.asarray(arr, np.float32)
                    elif dtype.startswith("int"):
                        arg = np.asarray(arr, np.int32)
                    elif dtype.startswith("bool"):
                        arg = np.asarray(arr)
                    else:
                        print("Unsupported dtype!!!!!!!!!!!!")
                    # print(arg.nbytes)
                    return cpu_backend.buffer_from_pyval(arg)
                else:
                    if isinstance(arr, int):
                        arg = np.asarray(arr, np.int32)
                        return cpu_backend.buffer_from_pyval(arg)
                    else:
                        print("Unsupported dtype!!!!!!!!!!!!")
                    return cpu_backend.buffer_from_pyval(arg)
            input_buffers = [to_buffer(arg) for arg in dyn_args]

            if time_record:
                    start_time = time.perf_counter()
            output_buffers = cpu_executable.execute(input_buffers)
            # if time_record:    
            #     end_time = time.perf_counter()
            #     print(f"Elapsed time for runtime: {end_time - start_time}")
            
            wrapper.pphlo = pphlo_cur
           
            # if time_record:
                    # start_time = time.perf_counter()
            output = [np.asarray(buf) for buf in output_buffers]
            if time_record:    
                end_time = time.perf_counter()
                print(f"Elapsed time for runtime: {end_time - start_time}")
                # print(f"Elapsed time for read buffer: {end_time - start_time}")

            if len(output) == 1:
                return output[0]
            else:
                return tuple(output)

        elif backend == "interpreter":
            with ir.Context() as context:
                stablehlo.register_dialect(context)
                curr_version = stablehlo.get_current_version()
                m = ir.Module.parse(pphlo_cur)
                args = []
                for dyn_arg in dyn_args:
                    dtype = str(dyn_arg.dtype)
                    if dtype.startswith("float"):
                        arg = np.asarray(dyn_arg, np.float32)
                    elif dtype.startswith("int"):
                        arg = np.asarray(dyn_arg, np.int32)
                    elif dtype.startswith("bool"):
                        arg = np.asarray(dyn_arg)
                    else:
                        print("Unsupported dtype!!!!!!!!!!!!")
                    args.append(ir.DenseFPElementsAttr.get(arg))
                if time_record:
                    start_time = time.perf_counter()
                output_tensor = stablehlo.eval_module(m, args)
                if time_record:    
                    end_time = time.perf_counter()
                    print(f"Elapsed time for runtime: {end_time - start_time}")
                output = []
                for tensor in output_tensor:
                    type_tensor = str(tensor.type)
                    if type_tensor.split("x")[-1].startswith("f"):
                        output.append(np.asarray(tensor, np.float32))
                    elif type_tensor.split("x")[-1].startswith("i"):
                        output.append(np.asarray(tensor, np.int32))
                    else:
                        print("Unsupported dtype!!!!!!!!!!!!")

            wrapper.pphlo = pphlo_cur
            if len(output) == 1:
                return output[0]
            else:
                return tuple(output)
            
        else:
            pphlo_cur_neglect = re.sub(r'dense<".*?">', 'dense<neglected>', pphlo_cur)
            pphlo_cur_neglect = re.sub(r'dense<\[\[.*?\]\]>', 'dense<neglected>', pphlo_cur_neglect)
            if pphlo_ref != (None, None):
                pphlo_cur_hash = hash(pphlo_cur_neglect.strip())
                pphlo_path = pphlo_ref[0]
                pass_option_cur = pphlo_ref[1]
                # implement write lock
                wrtie_lock_path = os.path.join(pphlo_path, "pphlo_" + fun.__name__ + ".lock")

                # get the write lock
                max_attempts = 5
                attempt = 0
                delay_seconds = 1
                while True:
                    if attempt >= max_attempts:
                        raise Exception(f"Could not get lock '{wrtie_lock_path}'")
                    if not os.path.exists(wrtie_lock_path):
                        try:
                            fd = os.open(wrtie_lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                            os.close(fd)
                            break
                        except:
                            print(f"Could not get lock '{wrtie_lock_path}'")
                            attempt += 1
                            continue
                    
                    time.sleep(delay_seconds)

                with open(os.path.join(pphlo_path, "pphlo_" + fun.__name__ + ".json"), "r") as file:
                    pphlo_record_dict = json.load(file)
                IR_existed = False
                for pass_option in list(pphlo_record_dict.keys()):
                    if pass_option_cur != pass_option and pphlo_record_dict[pass_option]["pphlo"] == pphlo_cur_hash:
                        pphlo_record_dict[pass_option]["deduplication"].append(pass_option_cur)
                        IR_existed = True
                        break
                if not IR_existed:
                    pphlo_record_dict[pass_option_cur] = {"pphlo": pphlo_cur_hash, "deduplication": [pass_option_cur]}
                with open(os.path.join(pphlo_path, "pphlo_" + fun.__name__ + ".json"), "w") as file:
                    json.dump(pphlo_record_dict, file)
                try:
                    os.remove(wrtie_lock_path)
                except Exception as e:
                    print(f"Could not release lock '{wrtie_lock_path}': {e}")
                if IR_existed:
                    print(f"IR is not changed for {fun.__name__}")
                    return "skipped"

            wrapper.pphlo = pphlo_cur
            out_flat = sim(executable, *args_flat)
            
            _, output_tree = jax.tree_util.tree_flatten(output)

            return jax.tree_util.tree_unflatten(output_tree, out_flat)

    return wrapper

def sim_hlo(
    sim: Simulator,
    IR: str,
    output_num: int,
    args: list, 
    copts=spu_pb2.CompilerOptions(),
):
    in_names = [f'in{idx}' for idx in range(len(args))]
    out_names = [f'in{idx}' for idx in range(output_num)]
    ir_text = xla.hlo_module_from_text(IR).as_serialized_hlo_module_proto()
    executable = spu_fe.compile_hlo(
        ir_text,
        [spu_pb2.Visibility.VIS_SECRET] * len(args),
        in_names,
        out_names,
        copts=copts,
    )
    # print(executable.code.decode("utf-8"))
    args_flat, _ = jax.tree_util.tree_flatten((args, {}))
    out_flat = sim(executable, *args_flat)
    return out_flat
