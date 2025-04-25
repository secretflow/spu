Please use `examples/python/conf/4pc.json` to start nodes for **outsourced** 4PC setting. Here two nodes of kind `PYU` play the role of clients to share data, while the four `SPU` nodes are the servers for 4PC. In `fantastic4/jmp.h`, you can define `OPTIMIZED_F4` to use our optimized protocols for some primitives.

## Run Experiment
To run experiments in `examples/python/ml`, you can start two terminals:
- In first terminal, launch 4pc SPU backend runtime
    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- -c examples/python/conf/4pc.json up
    ```
- In second terminal, run your exapmle, such as:

    ```sh
    bazel-bin/examples/python/ml/ss_xgb/ss_xgb -c examples/python/conf/4pc.json
    ```
    After running, you get output results like: 
    train time 8.816999197006226
    predict time 0.5341589450836182
    auc 0.9951574969610486

## Asymmetry Costs
Note that, in Fantastic4, each party has **asymmetry role** and thus different communication costs and computation.

To print the detailed costs of each party, it can help to:

- In `distributed_impl.py`, remove the restriction of `my_rank != 0` for the generation of profiles `spu_config.enable_hal_profile` and so on. You will see the logs of each party.

In some applications like NN, the whole task is split into hundreds of batches. Each batch is executed separately and the profiles do not accumulate across multiple subtasks.

- In `api.cc`, the function `printProfilingData` print overall cost information. You can similarly write and store the profiles of each party to your local files, which will help you to manually aggregate the the costs across multiple subtasks to obtain the total profiles of the whole task.
