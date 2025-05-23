# Performance for Shamir-based Protocols

**Note:**

The reported performance below reflects **online performance**, controlled by the `ONLINE_ONLY` flag in `src/libspu/mpc/experimental/shamir/arithmetic.h`

If you want to measure **total performance** (which is slower due to various reasons), you can disable this definition.

## ss_lr

**Scripts:**

```shell
# Start nodes with Shamir configuration
bazel run -c opt //examples/python/utils:nodectl -- -c examples/python/conf/3pc_shamir.json up

# Run the ss_lr script with Shamir configuration
bazel run -c opt //examples/python/ml/ss_lr -- -c examples/python/conf/3pc_shamir.json


```

Alternatively, you can run the node startup in the background:

```shell
bazel run -c opt //examples/python/utils:nodectl -- -c examples/python/conf/3pc_shamir.json up &
bazel run -c opt //examples/python/ml/ss_lr -- -c examples/python/conf/3pc_shamir.json
```

**Online Performance:**

- **LAN:**
  - Traning time: 49.7 seconds
  - Prediction time: 5 seconds
  - Accuracy: 88.27%
- **WAN** (RTT latency = 10ms, bandwidth = 100 Mbps):
  - Traning time: 476 seconds
  - Prediction time: 105 seconds
  - Accuracy: 88.27%

## stat_nn

**Scripts:**

```shell
# Start nodes with Shamir configuration
bazel run -c opt //examples/python/utils:nodectl -- -c examples/python/conf/3pc_shamir.json up

# Run the ss_lr script with Shamir configuration
bazel run -c opt //examples/python/ml/stax_nn -- -c examples/python/conf/3pc_shamir.json
```

Alternatively, you can run the node startup in the background:

```shell
bazel run -c opt //examples/python/utils:nodectl -- -c examples/python/conf/3pc_shamir.json up &
bazel run -c opt //examples/python/ml/stax_nn -- -c examples/python/conf/3pc_shamir.json
```

**Online Performance:**

The following numbers reflect the performance of a **single model update**:

- LAN: 1-2 seconds
- WAN (RTT latency = 10ms, bandwidth = 100 Mbps):
  - a updation: 26 seconds

## WAN Simulation

To simulate a WAN environment with **10ms RTT latency** and **100Mbps bandwidth** using the `tc` command,
configure the local network interface (lo) as follows:

Since both incoming and outgoing traffic experience latency, set the one-way delay to **5ms**.

```shell
sudo tc qdisc add dev lo root handle 1: htb default 10
sudo tc class add dev lo parent 1: classid 1:10 htb rate 100mbit
sudo tc qdisc add dev lo parent 1:10 handle 10: netem delay 5ms
```
