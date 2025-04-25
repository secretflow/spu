# Notes

Please use `examples/python/conf/4pc.json` to start nodes for **outsourced** 4PC setting.
Here two nodes of kind `PYU` play the role of clients to share data, while the four `SPU` nodes are the servers for 4PC.
In `fantastic4/jmp.h`, you can define `OPTIMIZED_F4` to use our optimized protocols for some primitives.

## Run Experiment

To run experiments in `examples/python/ml`, you can start two terminals:

- In first terminal, launch 4pc SPU backend runtime

    ```sh
    bazel run -c opt //examples/python/utils:nodectl -- -c examples/python/conf/4pc.json up
    ```

- In second terminal, run your example, such as:

    ```sh
    bazel run -c opt //examples/python/ml/ss_xgb:ss_xgb -- -c examples/python/conf/4pc.json
    ```

    After running, you get output results like:
    train time 8.816999197006226,
    predict time 0.5341589450836182,
    auc 0.9951574969610486

## Asymmetry Costs

Note that, in Fantastic4, each party has **asymmetry role** and thus different communication costs and computation.

To print the detailed costs of each party, it can help to:

- In `spu/utils/distributed_impl.py` Line 555,
remove the restriction of `my_rank != 0` for the generation of profiles `spu_config.enable_hal_profile` and so on. You will see the logs of each party.

In some applications like NN, the whole task is split into hundreds of batches.
Each batch is executed separately and the profiles do not accumulate across multiple subtasks.

- In `libspu/device/api.cc`, the function `printProfilingData` prints overall cost information.
You can similarly write and store the profiles of each party in your local files,
which will help you to manually aggregate the costs across multiple subtasks to obtain the total profiles of the whole task.
For example, we can use `ofstream` to write MPC profiles to a csv file:

    ```c++
    void printProfilingData(spu::SPUContext *sctx, const std::string &name,
                            const ExecutionStats &exec_stats,
                            const CommunicationStats &comm_stats) {
    // print overall information
    SPDLOG_INFO(
        "[Profiling] SPU execution {} completed, input processing took {}s, "
        "execution took {}s, output processing took {}s, total time {}s.",
        name, getSeconds(exec_stats.infeed_time),
        getSeconds(exec_stats.execution_time),
        getSeconds(exec_stats.outfeed_time), getSeconds(exec_stats.total_time()));

    // print action trace information
    std::filesystem::path curPath = std::filesystem::current_path();
    std::cout << "Current Path = " << curPath << std::endl;

    auto rank = sctx->lctx()->Rank();
    std::ofstream csv_file;
    std::string filename = "f4_lr_Party" + std::to_string((int64_t)rank) + ".csv";
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

    {
        std::map<ActionKey, ActionStats> stats;

        const auto &tracer = GET_TRACER(sctx);
        const auto &records = tracer->getProfState()->getRecords();

        for (const auto &rec : records) {
        auto &stat = stats[{rec.name, rec.flag}];
        stat.count++;
        stat.total_time +=
            std::chrono::duration_cast<Duration>(rec.end - rec.start);
        stat.send_bytes += (rec.send_bytes_end - rec.send_bytes_start);
        stat.recv_bytes += (rec.recv_bytes_end - rec.recv_bytes_start);
        stat.send_actions += (rec.send_actions_end - rec.send_actions_start);
        stat.recv_actions += (rec.recv_actions_end - rec.recv_actions_start);
        }

        static std::map<int64_t, std::string> kModules = {
            /*{TR_HLO, "HLO"}, {TR_HAL, "HAL"}, */{TR_MPC, "MPC"}};

        for (const auto &[mod_flag, mod_name] : kModules) {
        if ((tracer->getFlag() & mod_flag) == 0) {
            continue;
        }

        double total_time = 0.0;
        std::vector<ActionKey> sorted_by_time;
        for (const auto &[key, stat] : stats) {
            if ((key.flag & mod_flag) != 0) {
            total_time += stat.getTotalTimeInSecond();
            sorted_by_time.emplace_back(key);
            }
        }

        std::sort(sorted_by_time.begin(), sorted_by_time.end(),
                    [&](const auto &k0, const auto &k1) {
                    return stats.find(k0)->second.getTotalTimeInSecond() >
                            stats.find(k1)->second.getTotalTimeInSecond();
                    });

        SPDLOG_INFO("{} profiling: total time {}", mod_name, total_time);
        csv_file << "phase total" << ","
        << "1" << ","
        << total_time << ","
        << comm_stats.send_bytes << ","
        << comm_stats.recv_bytes << ","
        << comm_stats.send_actions << ","
        << comm_stats.recv_actions << "\n";
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
        }
    }

    // print link statistics
    SPDLOG_INFO(
        "Link details: total send bytes {}, recv bytes {}, send actions {}, recv "
        "actions {}",
        comm_stats.send_bytes, comm_stats.recv_bytes, comm_stats.send_actions,
        comm_stats.recv_actions);

    csv_file.close();
    }
    ```

After that, you can simply run examples as mentioned, collect detailed statistics of each operation,
and aggregating the communication costs and time corresponding to the key `phase total` to obtain the total results.
For example, we run `stax_nn` with default parameters using Fantastic4 in about 3050 seconds (aggregated `duration_seconds` of MPC profiles)
at the costs of 159850 MB communication in total (aggregate 4 parties' `send_bytes`),
and the precision is 0.9685.
For default `ss_lr`, it requires 75 seconds with 1437 MB and obtains auc_score 0.8827288523888388.
You can also use `tc` command to simulate different networks and compare the aggregated statistics.
