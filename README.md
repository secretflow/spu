# USENIX ATC '23 Artifact Evaluation

## 1. Overview
This branch contains code and document for reproducing experiments reported in our accepted USENIX ATC '23 paper.
The code in this branch is mainly based on SPU tag [0.3.2b1](https://github.com/secretflow/spu/releases/tag/v0.3.2b1) which was the stable version at the time of we submitting our paper.
As SPU is an open-source project under active development, the code in this branch may have a significant gap with the current main branch.

This README describes the artifact and provides a road map for evaluation.
For more details on this repo's directory layout, please refer to [REPO_LAYOUT.md](./REPO_LAYOUT.md).

## 2. System Requirements 
### Hardware
**SPU has no special hardware requirements.** 
We have done our evaluations on **three** Alibaba Cloud ecs.g7.xlarge cloud servers with 4 vCPU and 16GB RAM each. 
The three servers are connected with high-performance network (10.1Gbps bandwidth and 0.1ms round-trip time).
The CPU model is Intel(R) Xeon(R) Platinum 8369B CPU @ 2.70GHz.

### Software
We evaluated SPU on Ubuntu 20.04.5 LTS with Linux kernel 5.4.0-125-generic.
Technically, SPU is supported to run on any Linux servers with software requirements described in [CONTRIBUTING.md](./CONTRIBUTING.md#linux).
The testing servers should at least have `git` and `docker` installed as we provides users with a docker image which has pre-installed most dependencies of SPU.
The docker file is located at [atc23-ae.DockerFile](./docker/atc23-ae.DockerFile).

## 3. Getting Started

### 3.0 Check requirements

1. Make sure your servers has installed `git` and `docker` (Run on server0, server1, and server2):

    ```console
    $ git --version
    $ docker -v
    ```

2. Get your servers' network interface IP addresses which will be used in the following steps (Run on server0, server1, and server2):

    ```console
    $ ip addr
    ```
    The network interface names of your servers should be something like `eth0`.

### 3.1 Clone the AE branch

Run on server0, server1, and server2:
```console
$ git clone -b atc23_ae https://github.com/secretflow/spu.git
$ cd spu
```

### 3.2 Launch docker containers

Run on server0, server1, and server2:
```console
$ docker run -d -it --network host --name spu-ae-$(whoami) --mount type=bind,source="$(pwd)",target=/home/admin/dev/ -w /home/admin/dev --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=NET_ADMIN --privileged=true secretflow/atc23-ae:0.1
```

### 3.3 Enter docker containers 

Run on server0, server1, and server2:
```console
$ docker exec -it spu-ae-$(whoami) bash
```

### 3.4 Build SPU

**The following steps run inside docker containers**

Run on server0, server1, and server2:
```console
$ bazel build -c opt //examples/python/utils:nodectl
```
> **Note:**
> Running this step for the first time has to download dependencies from Internet and compile SPU from source code so it takes **considerable** time.

### 3.5 Configure SPU cluster

Run on server0, server1, and server2:

The default SPU runtime configuration is located at [3pc.json](examples/python/conf/3pc.json).
As you can see, we have defined five nodes in the file. 
Three of them (i.e., node0, node1, and node2) are SPU backend nodes which perform MPC computations.
The other two nodes (i.e., node3 and node4) act as data providers. 

We place `node0` on server0, `node1` and `node3` on server1, `node2` and `node4` on server2.
You should replace nodes' IPs with your servers' IPs in **each server's configuration**.
Both `nodes` and `spu_internal_addrs` fields should be modified.
Besides, you should guarantee that the specified ports are not used or blocked by your servers' network firewall.

### 3.6 Launch SPU cluster

Run on server0:
```console
$ nohup bazel run -c opt //examples/python/utils:nodectl -- start --node_id node:0 &> node0.log &
```

Run on server1:
```console
$ nohup bazel run -c opt //examples/python/utils:nodectl -- start --node_id node:1 &> node1.log &
$ nohup bazel run -c opt //examples/python/utils:nodectl -- start --node_id node:3 &> node3.log &
```

Run on server2:
```console
$ nohup bazel run -c opt //examples/python/utils:nodectl -- start --node_id node:2 &> node2.log &
$ nohup bazel run -c opt //examples/python/utils:nodectl -- start --node_id node:4 &> node4.log &
```

### 3.7 Check SPU cluster status

Run on server0:
```console
$ tail node0.log
```

Run on server1:
```console
$ tail node1.log
$ tail node3.log
```

Run on server2:
```console
$ tail node2.log
$ tail node4.log
```

If each node log prints something like blow, that means the SPU cluster has been launched successfully:
```console
[2023-05-09 10:46:00,098] [MainProcess] Starting grpc server at 172.19.245.237:9920
```

## 4. Run Experiments

### 4.1 Run neural network training experiments

1. Local Area Network (LAN)
   
    We provide a script `run-nn.sh` to run neural network training tasks with different settings. 

    Run on server0:
    ```console
    $ cat run-nn.sh
    for net in network_a network_b network_c network_d; do
        for opt in sgd amsgrad adam; do
            start_ts=$(date +%s)
            echo "Start training "${net}" "${opt}" "${start_ts}""
            bazel run -c opt //examples/python/ml:stax_nn -- --model ${net} --optimizer ${opt} &> ${net}-${opt}.log
            end_ts=$(date +%s)
            echo "Finish training "${net}" "${opt}" "${end_ts}""
            echo "Elapsed time: $[end_ts-start_ts]"
        done
    done

    $ nohup bash run-nn.sh &> spu.log &
    ```
    
    You can check the training status and results by:
    ```console
    $ tail network_a-sgd.log
    15:51:05.627001 Epoch: 5/5  Batch: 461/468
    15:51:05.752683 Epoch: 5/5  Batch: 462/468
    15:51:05.877300 Epoch: 5/5  Batch: 463/468
    15:51:06.000546 Epoch: 5/5  Batch: 464/468
    15:51:06.121920 Epoch: 5/5  Batch: 465/468
    15:51:06.250110 Epoch: 5/5  Batch: 466/468
    15:51:06.376858 Epoch: 5/5  Batch: 467/468
    15:51:06.496595 Epoch: 5/5  Batch: 468/468
    train(spu) elapsed time: 292.6806 seconds
    accuracy(spu): 0.9685
    ```

    > **Note:**
    > Running this step for the first time has to download datasets from Internet. It may take **several hours** to run all experimental settings. You can also choose to run each experimental setting individually.
    

2. Wide Area Network (WAN)

    Running training tasks under WAN is the same as under LAN except we should constrain network bandwidth and latency at first.
    We use the `tc` command to simulate the WAN environment.

    Run on server0, server1, and server2:
    ```console
    $ tc qdisc del dev eth0 root
    $ tc qdisc add dev eth0 root handle 1: tbf rate 300mbit burst 256kb latency 800ms                                  
    $ tc qdisc add dev eth0 parent 1:1 handle 10: netem delay 20msec limit 8000 
    ```

    After setting the network environment, you can run the training task again.

    Run on server0:
    ```console
    $ nohup bash run-nn.sh &> spu.log &
    ```

    > **Note:**
    > Training under WAN is far more slow. It may take several days to run all the experimental settings. You can simply verify SPU's performance results by running several iterations.
    
    **If you cancel the training task during running with `Crtl` + `c`, please kill the `nodectl` process on each server and launch the SPU cluster again.**
    
    Run on server0, server1, and server2:
    ```console
    $ kill -9 `ps -ef | grep nodectl | grep -v grep | awk '{print $2}'`
    ```

### 4.2 Run VAE training experiment

1. First, clean up the network constrains

    Run on server0, server1, and server2:
    ```console
    $ tc qdisc del dev eth0 root
    ```

2. Relaunch SPU cluster

    This experiment requires a higher precision setting than the default. Set `"fxp_fraction_bits": 24` in `3pc.json`, kill `nodectl` process, and relaunch SPU cluster.

3. Run `flax_vae` experiment

    Run on server0:
    ```console
    $ bazel run -c opt //examples/python/ml/flax_vae:flax_vae -- --output_dir `pwd` --num_epochs 5
    ```
    
    > **Note:**
    > Running this experiment for the first time has to download datasets from Internet and it takes nearly two and half hours to run the five epochs.


4. Check results

    When training is finished, you can check the generated images in the specified `output_dir` and compare the results of SPU and CPU versions.

### 4.3 Run LSTM training experiment

1. Install dependencies 

    Run on server0:
    ```console
    $ pip install -r examples/python/ml/haiku_lstm/requirements.txt
    ```

2. Relaunch SPU cluster

    This experiment requires a higher precision setting than the default. Set `"fxp_fraction_bits": 24` in `3pc.json` and relaunch SPU cluster.

3. Run `haiku_lstm` experiment

    Run on server0:
    ```console
    $ bazel run -c opt //examples/python/ml/haiku_lstm:haiku_lstm -- --output_dir `pwd`
    ```

    > **Note:**
    > Running this experiment for the first time has to download datasets from Internet and it takes nearly two hours to run 2000 steps.


4. Check results

    When training is finished, you can check the generated images in the specified `output_dir` and compare the results to CPU versions.

### 4.4 Run TensorFlow experiment

1. Relaunch SPU cluster:

    Set `"fxp_fraction_bits": 18` in `3pc.json` and relaunch SPU cluster.

2. Run experiment

    Run on server0:
    ```console
    $ bazel run -c opt //examples/python/ml:tf_experiment
    ```

### 4.5 Run PyTorch experiment

1. Install a third-party dependency [Torch-MLIR](https://github.com/llvm/torch-mlir).

    Run on server0:
    ```console
    $ pip install https://github.com/llvm/torch-mlir/releases/download/snapshot-20220830.581/torch-1.13.0.dev20220830+cpu-cp38-cp38-linux_x86_64.whl
    $ pip install https://github.com/llvm/torch-mlir/releases/download/snapshot-20220830.581/torch_mlir-20220830.581-cp38-cp38-linux_x86_64.whl
    ```

2. Run experiment

    Run on server0:
    ```console
    $ bazel run -c opt //examples/python/ml/torch_experiment:torch_experiment
    ```
