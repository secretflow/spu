This repo stores the implementation code of Alkaid.

## Usage

```shell
# microbenchmark from cpp side.
bazel build //libspu/mpc:mfippa_test --jobs 32
./bazel-out/k8-fastbuild/bin/libspu/mpc/mfippa_test

# activation function benchmark from python side.
bazel build //examples/python/utils:nodectl --jobs 32
bazel build //examples/python/ml/flax_gpt2:albenchmark --jobs 32
./bazel-bin/examples/python/utils/nodectl -c examples/python/ml/flax_gpt2/alkaid.json up
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/albenchmark -c examples/python/ml/flax_gpt2/3pc.json
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/albenchmark -c examples/python/ml/flax_gpt2/alkaid.json

# nn inference benchmark from python side.
bazel build //examples/python/utils:nodectl --jobs 32
bazel build //examples/python/ml/flax_gpt2:pumabench --jobs 32
./bazel-bin/examples/python/utils/nodectl -c examples/python/ml/flax_gpt2/alkaid.json up
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/pumabench -c examples/python/ml/flax_gpt2/3pc.json
./bazel-out/k8-fastbuild/bin/examples/python/ml/flax_gpt2/pumabench -c examples/python/ml/flax_gpt2/alkaid.json
```