# Nimbus: Secure and Efficient Two-Party Inference for Transformers
---
This branch contains the [TODO (NIPS 2024)](TODO)'s implementation of the linear protocol. The implementation is based on the MPC backend contained in /state/partition/zyli/nimbus_opensource/spu/libspu/mpc/cheetah, which is the official codes of [BumbleBee](https://eprint.iacr.org/2023/1678). The nonlinear implementation can directly use python frontend of the SecretFlow.

## Setup the Environment

Use the official docker image from SecretFlow. The implemention is based on an old commit of the SecretFlow/SPU. Using new version of the docker image may raise unexpected error when using the bazel to build.
```bash
docker pull secretflow/release-ci:20231208
```

Create the container using the following commands.
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

Enter the container and install the necessary python libraries.
```bash
pip install numpy==1.25.2
pip install multiprocess==0.70.15
pip install cloudpickle==2.2.1
pip install grpcio==1.59.0
pip install jax==0.4.14
pip install jaxlib==0.4.14
pip install google
pip install google-cloud
pip install google-cloud-vision
pip install termcolor
```

<!-- ## Fix the disalignment of the YACL checksum
Go to "bazel/repositories.bzl" find the commit id of the used YACL repository. Then checkout the corresponding commit of the YACL repository. Replace the wrong checksum (sha256) "1ecfa70328221748ceb694debffa0106b92e0f9bf6a484f8e8512c2730c7d730" by the "d70f42832337775edb022ca8ac1ac418f272e791ec147778ef7942aede414cdc". Also replace the strip_prefix "ipp-crypto-ippcp_2021.8" by "cryptography-primitives-ippcp_2021.8". Then, generate a patch file and put it under "bazel/patches". and Change the "patch_args" and "patches" in "bazel/repositories.bzl". -->

## Run the Protocol
In one container, move to the root path of spu. Start the service using the following command. The "nodes" and "spu_internal_addrs" in the json file should be the IP address of your container.
```bash
bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/Nimbus/2pc.json up
```

In another container, run the code. The four numbers indicates shapes of a batched matrix multiplication (b, m, k)*(b, k, n), where the former is the weight tensor and the later is the activation tensor. Variables m, n, k are out_dim, hidden_dim, token_num. 
```bash
bazel run -c opt examples/python/ml/Nimbus/linear 1 3072 768 128
```

## Switch between Correctness Verification and Time Profiling 
We separate the correctness verification and time profiling of the implemention. This is because Nimbus requires a setup stage that cache the encrypted model weights at the client. However, the SecretFlow framework does not have the concept of the setup stage and all operations are conducted at the online stage. Switching correctness verification and time profiling is associated with the varable "profiling" at around the 321-th line in libspu/mpc/cheetah/arithmetic.cc.

* profiling=true; 
The default value of profiling is set as true. This mode only verifies the correctness of the protocol implemention. The above two commands are supposed to generate the correct results of the batched matrix multiplication. This mode requires more time to execute since the model weights need to be sent online.
* profiling=false; 
To obatin the performance, you may change the "profiling" as false and re-run the above two commands. In such case, a random encrytped weights are generated for execution instead of sending. This simulates the client already has the encrypted weights in the memory. Then the expected running time is obtained.