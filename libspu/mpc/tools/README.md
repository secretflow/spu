# Run benchmark

## Basic Usage

You can get help info like this:

```sh
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
  --numel=<uint>          - number of benchmark elements, default: [2^10, 2^20]
  --parties=<string>      - server list, format: host1:port1[,host2:port2, ...]
  --protocol=<string>     - benchmark protocol, supported protocols: semi2k / aby3, default: aby3
  --rank=<uint>           - self rank, starts with 0
  --shiftbit=<uint>       - benchmark shift bit, default: 2

Generic Options:

  --help                  - Display available options (--help-hidden for more)
  --help-list             - Display list of available options (--help-list-hidden for more)
  --version               - Display the version of this program

```

Because many parameters have default parameters, we don't have to specify each parameter.
If you want run it with **standalone mode**, you can do this:

```sh
bazel run -c opt libspu/mpc/tools/benchmark -- --benchmark_counters_tabular=true
```

If you want **mparty mode** on localhost, you need start multi processes to simulate different parties, and we only care output of rank 0.
eg: run **aby3** **mparty** benchmark as follows, you can create a script.:

```sh
$ cat run_mpart_bench.sh
#!/bin/sh
bazel run -c opt //libspu/mpc/tools/benchmark -- --mode=mparty --rank=1 2>&1 >/dev/null &
bazel run -c opt //libspu/mpc/tools/benchmark -- --mode=mparty --rank=2 2>&1 >/dev/null &
bazel run -c opt //libspu/mpc/tools/benchmark -- --benchmark_counters_tabular=true --mode=mparty --rank=0
$ sh run_mpart_bench.sh
```

or

```sh
sh docs/reference/run_benchmark.sh --output LAN.json --mode mparty
```

## Format benchmark output

You can use **--benchmark_out=*.json --benchmark_out_format=json** to specify the output json or **docs/reference/run_benchmark.sh**, eg:

```sh
$ bazel run -c opt //libspu/mpc/tools:benchmark -- --mode=standalone \
  --benchmark_out=standalone.json --benchmark_out_format=json
```

or

```sh
sh docs/reference/run_benchmark.sh --output standalone.json
```

## multi-party with network limitations

**docs/reference/run_benchmark.sh** is recommend.
To set network limitation manually, execute following commands on each docker or machine,
then specify ip with `parties` parameter, eg:
set 300Mbps bandwidth，add 20ms latency:

```sh
$ tc qdisc del dev eth0 root
$ tc qdisc add dev eth0 root handle 1: tbf rate 300mbit burst 256kb latency 800ms
$ tc qdisc add dev eth0 parent 1:1 handle 10: netem delay 20msec limit 8000
$ bazel run -c opt //libspu/mpc/tools/benchmark -- \
   --parties="172.17.0.13:9540,172.17.0.14:9541,172.17.0.15:9542" \
   --benchmark_counters_tabular=true --mode=mparty --rank=1
```

## Usage of docs/reference/run_benchmark.sh

This script can help to generate json output for different mode.
You can always specify output json filename with **--output**.

### - standalone

```sh
sh docs/reference/run_benchmark.sh
```

This will generate standalone.json.

### - mparty without network limit

For example, you can run this on 172.17.0.8, and use 172.17.0.1 as rank 1&2.
This outputs WAN.json. Before run this script, you need **ssh-copy-id** to remote machine.

```sh
sh docs/reference/run_benchmark.sh --mode mparty --remote huocun@172.17.0.1 --remote_dir /home/ssd0/huocun/ppu --remote_python_env /home/ssd0/huocun/ppu/venv/bin/activate --parties 172.17.0.8:9444,172.17.0.1:9445,172.17.0.1:9446 --output LAN.json
```

### - mparty with network limit

Based on 5.2, you can use **--rate** and **--delay** to specify networt rate and **add delay**.
New round-trip delay will be old round-trip delay add delay values you set.
Tool tc needs to be installed. (You can run `yum install -y iproute-tc` to install tc.)
This will generate WAN_300mbit_20msec.json.

```sh
sh docs/reference/run_benchmark.sh --mode mparty --remote huocun@172.17.0.1 --remote_dir /home/ssd0/huocun/ppu --remote_python_env /home/ssd0/huocun/ppu/venv/bin/activate --parties 172.17.0.8:9444,172.17.0.1:9445,172.17.0.1:9446 --rate 300mbit --delay 20msec
```

## generate report

After we have generated a bunch of benchmark jsons,
we can use **docs/reference/gen_benchmark_report.py** to generate a report.
eg:

```python3
python docs/reference/gen_benchmark_report.py --output=report.md
    --input=LAN.json,../WAN_300mbit_20msec.json,WAN__.json
    --columns=env,buf_len
    --rows=op_name,field_type
    --values=time
    --sheet="Benchmark Protocol"
```

This command can generate a xlsx report(/csv/md/html is also supported), rows will be grouped by **op_name** and **field_type**,
columns with **env**, **buf_len**.
**env** is from **input** filenames, add different protocol will be in different sheets when we specify
**--output=*.md**.
**time** is formated by reasonable unit, raw data is "real_time".

'''

aby3

|                      |   ('time', 'LAN', '1024/us') |   ('time', 'LAN', '1048576/ms') |   ('time', 'WAN_300mbit_20msec', '1024/us') |   ('time', 'WAN_300mbit_20msec', '1048576/ms') |   ('time', 'WAN__', '1024/us') |   ('time', 'WAN__', '1048576/ms') |
|:---------------------|-----------------------------:|--------------------------------:|--------------------------------------------:|-----------------------------------------------:|-------------------------------:|----------------------------------:|
| ('add_sp', '128')    |                         9.38 |                            4.21 |                                       20.88 |                                           4    |                          13.35 |                              4.1  |
| ('add_sp', '64')     |                        11.02 |                            2.19 |                                       19.21 |                                           1.81 |                           9.57 |                              1.98 |
| ('add_ss', '128')    |                         9.46 |                            4.55 |                                       19.3  |                                           4.34 |                          13.41 |                              4.36 |
| ('add_ss', '64')     |                        12.56 |                            2.82 |                                       18.69 |                                           1.98 |                          15.96 |                              2.93 |

'''
