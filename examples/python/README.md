# Welcome to SPU python examples

To run a certain example, we have to first specify the distributed environment with:

> bazel run -c opt //examples/python/utils:nodectl -- up

To use a specific layout configuration (i.e. change MPC protocol, change outsourcing/colocated), change the configuration file accordingly.

> bazel run -c opt //examples/python/utils:nodectl -- -c examples/python/conf/2pc.json up

Then please check the comment of each example to run.

## Examples for PSI/PIR

Please check tests at this moment:

- spu/tests/legacy_psi_test.py
- spu/tests/psi_test.py
- spu/tests/ub_psi_test.py
- spu/tests/pir_test.py
