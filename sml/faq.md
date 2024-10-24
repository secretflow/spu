# FAQ

1. How can I know the supported operations in SPU?

    **Ans**: check [jax numpy status](https://www.secretflow.org.cn/docs/spu/latest/en-US/reference/np_op_status) or [xla status](https://www.secretflow.org.cn/docs/spu/latest/en-US/reference/xla_status);
    or you can direct test it using simulator or emulator.

2. How to adjust **field** and **fxp_fraction_bits** to improve precision?

   **Ans**: For simulator, you can create a `RuntimeConfig` Object and then pass to `Simulator`.

    ```python
    # for simulator
    config = spu_pb2.RuntimeConfig(
            protocol=spu_pb2.ProtocolKind.ABY3,
            field=spu_pb2.FieldType.FM128,  # change filed size here
            fxp_fraction_bits=30,  # change fxp here
        )
    sim = spsim.Simulator(3, config)
   ```

   For emulator, you can define a config, e.g. named `spu_128.json`,

   ```json
   {
    "id": "outsourcing.3pc",
    "nodes": {
        "node:0": "127.0.0.1:9920",
        "node:1": "127.0.0.1:9921",
        "node:2": "127.0.0.1:9922",
        "node:3": "127.0.0.1:9923",
        "node:4": "127.0.0.1:9924"
    },
    "devices": {
        "SPU": {
            "kind": "SPU",
            "config": {
                "node_ids": [
                    "node:0",
                    "node:1",
                    "node:2"
                ],
                "spu_internal_addrs": [
                    "127.0.0.1:9930",
                    "127.0.0.1:9931",
                    "127.0.0.1:9932"
                ],
                "runtime_config": {
                    "protocol": "ABY3",
                    "field": "FM128",
                    "fxp_fraction_bits": 30,
                    "enable_pphlo_profile": true,
                    "enable_hal_profile": true
                }
            }
        },
        "P1": {
            "kind": "PYU",
            "config": {
                "node_id": "node:3"
            }
        },
        "P2": {
            "kind": "PYU",
            "config": {
                "node_id": "node:4"
            }
        }
    }
   }
   ```

   Then, in python file, you can set up an emulator,

   ```python
   conf_path = "spu_128.json"  # path of json file defined above
   mode = emulation.Mode.MULTIPROCESS
   emulator = emulation.Emulator(conf_path, mode, bandwidth=300, latency=20)
   emulator.up()
   ```

3. Why the program I write runs correctly in plaintext, but behaves differently under MPC?

    **Ans**: it depends.
      - Huge error: you can check whether **overflow/underflow** is happened(often occurs when linear algebra ops such as `jax.numpy.linalg.*` are used),
       whether incidentally use **floating-point random generator** in SPU.
      - Mild error: you can switch to **larger ring** and more **fxp**.
