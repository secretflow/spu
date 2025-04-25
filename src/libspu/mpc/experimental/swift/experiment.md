# Experiment

### 1. Accuracy

#### KMEANS:

We set the "runtime_config" in `examples/python/config/3pc.json` as "SWIFT", and run:

`bazel run -c opt //examples/python/utils:nodectl -- up`

`bazel run -c opt //examples/python/ml/jax_kmeans:jax_kmeans`

**experimental result:**

```
Run on CPU
------

An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
[[-2.535751   2.1222713]
 [ 4.589908  -4.5871696]]
[1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 1. 1. 1. 1. 1. 0.
 0. 1. 1. 0. 1. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0.
 0. 0. 1. 0. 1. 0. 0. 0. 1. 1. 1. 1.]
Run on SPU
------

[[-2.5357475  2.1222687]
 [ 4.589905  -4.5871506]]
[1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 1. 1. 1. 1. 1. 0.
 0. 1. 1. 0. 1. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0.
 0. 0. 1. 0. 1. 0. 0. 0. 1. 1. 1. 1.]
```



#### ss_lr:

We set the "runtime_config" in `examples/python/config/3pc.json` as "SWIFT"，and set the "n_samples" in `examples/python/config/ds_mock_regression_basic.json` as 50000, and run:

`bazel run -c opt //examples/python/utils:nodectl -- up`

`bazel run -c opt //examples/python/ml/ss_lr:ss_lr`

**experimental result:**

```
[SWIFT]
train time 244.70280480384827
predict time 11.143281698226929
auc 0.8929411327226555

[ABY3]
train time 7.388535022735596
predict time 1.7024931907653809
auc 0.8929408743216628

```



#### stan_nn

We set the "runtime_config" in `examples/python/config/3pc.json` as "SWIFT", modify the `secureml()` in `examples/python/ml/stax_nn/models.py` as (you can skip this step if your computer has enough memory for the full connect layer with dimension 128):

```
def secureml():
    nn_init, nn_apply = stax.serial(
        Flatten,
        Dense(64),
        Relu,
        Dense(64),
        Relu,
        Dense(10),
    )
    return nn_init, nn_apply
```

and run:

`bazel run -c opt //examples/python/utils:nodectl -- up`

`bazel run -c opt //examples/python/ml/stax_nn:stax_nn`

**accuracy:**

We test the accuracy of CPU and SPU with epoch=1 and epoch=2, the accuracy of SPU is lower than CPU, we speculate that this is because SWIFT use the truncation from ABY3(https://dl.acm.org/doi/10.1145/3243734.3243760). The truncation method in ABY3 generate the share of $(r,rd=r/2^d)$, and calculate $x/2^d$ by $(x-r)/2^d+[rd]$, for a secret value $x \in [0, 2^{l_x}) \cup (2^l -2^{l_x},2^l)$，this method may cause a large error with the probability of $2^{-(l-l_x-1)}$. You can read Bicoptor2.0 (http://arxiv.org/abs/2309.04909) for detail.

| epoch |  CPU   |  SPU   |
| :---: | :----: | :----: |
|   1   | 91.66% | 88.25% |
|   2   | 93.77% | 90.01% |



### 2. Runtime

We test the runtime of SWIFT for ss_lr and stax_nn in LAN and WAN.

For LAN test, we use three local addresses as spu_internal_addrs (default setting in `examples/python/config/3pc.json`)

For WAN test, we use the `tc` tool in linux to emulate the WAN setting: `sudo tc qdisc add dev lo root netem rate 500Mbit delay 10ms`

#### ss_lr

The setting of runtime test on ss_lr follows accuracy test.

**result: **

|      | training | predict |
| :--: | :------: | :-----: |
| LAN  | 244.7 s  | 11.1 s  |
| WAN  | 1476.1 s | 46.6 s  |



#### stax_nn

The setting of runtime test on stax_nn follows accuracy test.

**result**

|      | runtime |
| :--: | :-----: |
| LAN  | 11429 s |
| WAN  | 56371 s |

