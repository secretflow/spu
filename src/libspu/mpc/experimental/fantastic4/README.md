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

- In `spu/utils/distributed_impl.py` Line 555, remove the restriction of `my_rank != 0` for the generation of profiles `spu_config.enable_hal_profile` and so on. You will see the logs of each party.

In some applications like NN, the whole task is split into hundreds of batches. Each batch is executed separately and the profiles do not accumulate across multiple subtasks.

- In `libspu/device/api.cc`, the function `printProfilingData` print overall cost information. You can similarly write and store the profiles of each party to your local files, which will help you to manually aggregate the the costs across multiple subtasks to obtain the total profiles of the whole task.
For example, we can use `ofstream` to write MPC profiles to csv:
```
    ...
    auto rank = sctx->lctx()->Rank();
    std::ofstream csv_file;
    std::string filename = "your_path/proto_example_Party" + std::to_string((int64_t)rank) + ".csv";
    if(!std::filesystem::exists(filename)){
        csv_file.open(filename);
        if (!csv_file.is_open()) {
        printf("dont open stats csv\n");
        }
        csv_file << "name,executed_times,duration_seconds,send_bytes,recv_bytes,send_actions,recv_actions\n";
    }
    else {
        csv_file.open(filename, std::ios::app);
    }
    ...
    ...
    ...
    for (const auto &key : sorted_by_time) {
    const auto &stat = stats.find(key)->second;
    SPDLOG_INFO(
        "- {}, executed {} times, duration {}s, send bytes {} recv "
        "bytes {}, send actions {}, recv actions {}",
        key.name, stat.count, stat.getTotalTimeInSecond(), stat.send_bytes,
        stat.recv_bytes, stat.send_actions, stat.recv_actions);
    csv_file << key.name << ","
    << stat.count << ","
    << stat.getTotalTimeInSecond() << ","
    << stat.send_bytes << ","
    << stat.recv_bytes << ","
    << stat.send_actions << ","
    << stat.recv_actions << "\n";
    }
    ...
```
After that, you can simply run examples as mentioned, collect detailed statistics of each operation, and aggregating the communication costs and time corresponding to the key `phase total` to obtain the total results. For example, we run `stax_nn` with default parameters using Fantastic4 in about 3050 seconds (aggregated `duration_seconds` of MPC profiles) at the costs of 159850
MB communication in total (aggregate 4 parties' `send_bytes`), and the precision is 0.9685. For default `ss_lr`, it requires 75 seconds with 1437 MB and obtains auc_score 0.8827288523888388. You can also use `tc` command to simulate different networks and compare the aggregated statistics.

