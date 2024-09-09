```shell
## start container
docker run -d -it --name spu-dev-cqy --mount type=bind,source="/home/chenxudong/workstation",target=/home/admin/dev/ -w /home/admin/dev --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=NET_ADMIN --privileged=true --entrypoint="bash" --gpus all secretflow/ubuntu-base-ci:latest

bazel build //libspu/mpc/aby3:mfippa_test --jobs 32
./bazel-out/k8-fastbuild/bin/libspu/mpc/aby3/mfippa_test

bazel build //libspu/mpc/alkaid:mfippa_test --jobs 32
./bazel-out/k8-fastbuild/bin/libspu/mpc/alkaid/mfippa_test

# 添加新安全计算框架后重新编译。其他时候无需编译。
bazel build //examples/python/utils:nodectl --jobs 32
# or: ./bazel-bin/examples/python/utils/nodectl up
./app/nodectl-app/nodectl up

bazel build //examples/python/ml/flax_mlp:flax_mlp --jobs 32
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_mlp/flax_mlp

bazel build //examples/python/ml/flax_gpt2:flax_gpt2 --jobs 32
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/flax_gpt2

bazel build //examples/python/ml/flax_gpt2:pumabench --jobs 32
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/pumabench

bazel build //libspu/mpc:mfippa_test --jobs 32
./bazel-out/k8-fastbuild/bin/libspu/mpc/mfippa_test

export http_proxy=http://192.168.109.37:7890
export https_proxy=http://192.168.109.37:7890

unset http_proxy 
unset https_proxy

./bazel-bin/examples/python/utils/nodectl -c examples/python/ml/flax_gpt2/3pc.json up
./bazel-bin/examples/python/utils/nodectl -c examples/python/ml/flax_gpt2/alkaid.json up

./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/flax_gpt2 -c examples/python/ml/flax_gpt2/3pc.json
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/flax_gpt2 -c examples/python/ml/flax_gpt2/alkaid.json

./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/primitive_test -c examples/python/ml/flax_gpt2/3pc.json
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/primitive_test -c examples/python/ml/flax_gpt2/alkaid.json

./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/pumabench -c examples/python/ml/flax_gpt2/3pc.json
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/pumabench -c examples/python/ml/flax_gpt2/alkaid.json

nohup ./bazel-bin/examples/python/utils/nodectl  -c examples/python/ml/flax_gpt2/3pc.json up > aby3.out 2>&1 &
nohup ./bazel-bin/examples/python/utils/nodectl -c examples/python/ml/flax_gpt2/alkaid.json up > alkaid.out 2>&1 &
nohup ./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/pumabench -c examples/python/ml/flax_gpt2/3pc.json >/dev/null 2>&1 &
nohup ./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/flax_gpt2 -c examples/python/ml/flax_gpt2/alkaid.json >/dev/null 2>&1 &
```

0. my view: 动态比特向量容器
1. 实现A2B双变体                              PPA 对比
2. 最高有效位提取我们的实现 的效率              MSB 对比ABY 2.0的电路，或者对比PPA 删掉一部分导线
3. 神经网络 算术电路不变，布尔电路修改          
4. 算术布尔乘法                               确认一下PPA最后的乘法能不能和bx融合
5. 多项式近似                                 布尔 and 布尔 and 算术


0. check aby3
1. check motion bit2a
2. a2b dynamic
3. bit2a ([x] -> [x^rb] -> y=x^rb -> y*ra)
4. eqz 

0. 128 bit ppa
1. 61 vs 62 vs 63
2. offline