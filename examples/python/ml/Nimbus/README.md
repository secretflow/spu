# Nimbus: Secure and Efficient Two-Party Inference for Transformers (NIPS 2024)


---
This branch contains the implementation of [Nimbus](https://nips.cc/virtual/2024/poster/95926) for linear and nonlinear layers. 

## Setup the Environment
We use the official Docker image from SecretFlow. This implementation is based on an old commit of the SecretFlow/spu. Using a newer version of the Docker image may result in unexpected errors when using Bazel to build.
```bash
docker pull secretflow/release-ci:20231208
```

Create the container using the following commands:
```bash
#!/bin/bash
docker_image=secretflow/release-ci:20231208

docker run -d -it --name <name> --rm \
--cap-add=NET_ADMIN \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--mount type=bind,source=<path_to_repo>,target=<path_to_repo_in_container> \
--cpuset-cpus="0-63" \
$docker_image
```

Once inside the container, install the necessary Python libraries:
```bash
pip install numpy==1.25.2
pip install multiprocess==0.70.15
pip install cloudpickle==2.2.1
pip install grpcio==1.59.0
pip install flax==0.7.4
pip install jax==0.4.14
pip install jaxlib==0.4.14
pip install google
pip install google-cloud
pip install google-cloud-vision
pip install termcolor
pip install matplotlib
```

## Run the Protocol
### Start the Service
In one container, navigate to the root path of SPU and start the service using the following command. The "nodes" and "spu_internal_addrs" in the JSON file `2pc.json` should be the IP addresses of your container:
```bash
bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/Nimbus/2pc.json up
```

### Test the Linear Layers
The implementation of the linear layer is based on the MPC backend called `cheetah`, which is the official code for [BumbleBee](https://eprint.iacr.org/2023/1678). The implementation can be found in the directory `/state/partition/zyli/nimbus_opensource/spu/libspu/mpc/cheetah`. 

To test the linear protocol, run the following command in another container. The four numbers represent the shapes of a batched matrix multiplication (b, m, k)*(b, k, n), where the former is the weight tensor and the latter is the activation tensor. The variables m, n, and k correspond to out_dim, hidden_dim, and token_num, respectively.

```bash
bazel run -c opt examples/python/ml/Nimbus/linear 1 3072 768 128
```

Switching between Correctness Verification and Time Profiling: We separate the correctness verification and time profiling of the implementation. This is necessary because Nimbus requires a setup stage that caches the encrypted model weights on the client. However, the SecretFlow framework does not incorporate the concept of a setup stage, and all operations are conducted in the online stage. Switching between correctness verification and time profiling is controlled by the variable "profiling," which can be found around the 321st line in `libspu/mpc/cheetah/arithmetic.cc`.

* `profiling=true;` 
The default value of profiling is set to true. In this mode, only the correctness of the protocol implementation is verified. The two commands above are expected to yield the correct results for batched matrix multiplication. This mode requires more time to execute since the model weights must be sent online.

* `profiling=false;` 
To obtain performance metrics, change the value of "profiling" to false and re-run the commands above. In this case, random encrypted weights are generated for execution instead of sending the weights. This simulates the scenario where the client already has the encrypted weights in memory, which results in obtaining the expected running time.

### Test the Nonlinear Layers
The nonlinear layer's implementation utilizes the Python frontend of SecretFlow. 
To evaluate the performance of nonlinear layers, run the following commands in another container. 

```bash
bazel run -c opt examples/python/ml/Nimbus/nonlinear -- --b 1 --token_num 128 --test_oursfake_softmax
bazel run -c opt examples/python/ml/Nimbus/nonlinear -- --b 1 --token_num 128 --test_bumblebeefake_softmax

bazel run -c opt examples/python/ml/Nimbus/nonlinear -- --b 1 --token_num 128 --test_oursfake_exp
bazel run -c opt examples/python/ml/Nimbus/nonlinear -- --b 1 --token_num 128 --test_bumblebeefake_exp

bazel run -c opt examples/python/ml/Nimbus/nonlinear -- --b 1 --token_num 128 --hidden 768 --test_oursfake_gelu
bazel run -c opt examples/python/ml/Nimbus/nonlinear -- --b 1 --token_num 128 --hidden 768 --test_bumblebeefake_gelu
```
To use a smaller ring and lower precision, change `"field": "FM64"` to `"field": "FM32"` and add `"fxp_fraction_bits": 13` in `/examples/python/ml/Nimbus/2pc.json`. The coefficients used in the piecewise polynomial are examples. This repository only profiles the running time. To test accuracy, you can generate the piecewise polynomial for each nonlinear layer and follow the fine-tuning codes in the [MPCFormer](https://github.com/DachengLi1/MPCFormer) library.