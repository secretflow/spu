This repo stores the implementation code of Alkaid.

## Usage

```shell
# microbenchmark from cpp side.
bazel build //examples/alkaid/benchmark:microbm --jobs 32
./bazel-bin/examples/alkaid/benchmark/microbm

# activation function benchmark from python side.
bazel build //examples/alkaid/utils:nodectl --jobs 32
bazel build //examples/alkaid/benchmark:actbm --jobs 32
./bazel-bin/examples/alkaid/utils/nodectl -c examples/alkaid/conf/alkaid.json up
./bazel-bin/examples/alkaid/benchmark/actbm -c examples/alkaid/conf/3pc.json
./bazel-bin/examples/alkaid/benchmark/actbm -c examples/alkaid/conf/alkaid.json

# gpt2 inference benchmark from python side.
bazel build //examples/alkaid/utils:nodectl --jobs 32
bazel build //examples/alkaid/benchmark:pumabm --jobs 32
./bazel-bin/examples/alkaid/utils/nodectl -c examples/alkaid/conf/alkaid.json up
./bazel-bin/examples/alkaid/benchmark/pumabm -c examples/alkaid/conf/3pc.json
./bazel-bin/examples/alkaid/benchmark/pumabm -c examples/alkaid/conf/alkaid.json
```