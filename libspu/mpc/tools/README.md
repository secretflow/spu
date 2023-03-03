# Run benchmark

## 1. Basic Usage

You can get help info like this:

```
$ bazel run -c opt libspu/mpc/tools/benchmark -- --help
USAGE: benchmark [options]

OPTIONS:
...
General options:

  --benchmark_**=<string> - google benchmark options, eg:
                                --benchmark_out=<filename>,
                                --benchmark_out_format={json|console|csv},
                                --benchmark_filter=<regex>,
                                --benchmark_counters_tabular = true,
                                --benchmark_time_unit={ns|us|ms|s}
  --mode=<string>         - benchmark mode : standalone / mparty, default: standalone
  --numel=<uint>          - number of benchmark elements, default: [10, 100, 1000]
  --parties=<string>      - server list, format: host1:port1[,host2:port2, ...]
  --party_num=<uint>      - server numbers
  --protocol=<string>     - benchmark protocol, supported protocols: semi2k / aby3, default: aby3
  --rank=<uint>           - self rank, starts with 0
  --shiftbit=<uint>       - benchmark shift bit, default: 4, 8]

Generic Options:

  --help                  - Display available options (--help-hidden for more)
  --help-list             - Display list of available options (--help-list-hidden for more)
  --version               - Display the version of this program

```

Because many parameters have default parameters, we don't have to specify each parameter.
If you want run it with **standalone mode**, you can do this:

```bash
bazel run -c opt libspu/mpc/tools/benchmark -- --benchmark_counters_tabular=true
```

If you want **mparty mode** on localhost, you need start `party_num` processes to simulate parties.
eg: run **aby3** **mparty** benchmark as follows, you can create a script.:

```
$ cat run_mpart_bench.sh
#!/bin/sh
bazel run -c opt //libspu/mpc/tools/benchmark -- --benchmark_counters_tabular=true --mode=mparty --rank=1 2>&1 >/dev/null &
bazel run -c opt //libspu/mpc/tools/benchmark -- --benchmark_counters_tabular=true --mode=mparty --rank=2 2>&1 >/dev/null &
bazel run -c opt //libspu/mpc/tools/benchmark -- --benchmark_counters_tabular=true --mode=mparty
$ sh run_mpart_bench.sh
```

## 2. Format benchmark output

Refer to [google benchmark](https://github.com/google/benchmark/blob/main/docs/user_guide.md),
`--benchmark_out="filename"` and `--benchmark_out_format={json|console|csv}` can be used to write benchmark output to a file, like this:

```
$ bazel run -c opt //libspu/mpc/tools/benchmark -- --protocol=aby3 \
    --benchmark_counters_tabular=true \
    --benchmark_out=$(pwd)/spu_mpc_benchmark_$(date +%F-%H-%M-%S).json \
    --benchmark_out_format=json
```

## 3. multi-party with network limitations

To test network limitations, execte follow commands on each docker or machine,
 then specify ip with `parties` parameter, eg:
set 300Mbps bandwidth，20ms latency:

```
$ tc qdisc del dev eth0 root
$ tc qdisc add dev eth0 root handle 1: tbf rate 300mbit burst 256kb latency 800ms
$ tc qdisc add dev eth0 parent 1:1 handle 10: netem delay 20msec limit 8000
$ bazel run -c opt //libspu/mpc/tools/benchmark -- \
   --parties="172.17.0.13:9540,172.17.0.14:9541,172.17.0.15:9542" \
   --benchmark_counters_tabular=true --mode=mparty --rank=1
```
