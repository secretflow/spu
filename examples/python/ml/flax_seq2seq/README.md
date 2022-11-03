## seq2seq addition example
This example is to demonstrate how to run legacy jax code (with minor modifications) on SPU. 

Thanks to SPU, we can run existing machine learning programs in a privacy-preserving way with ease-of-use. 

The seq2seq example comes from flax official github repo:

https://github.com/google/flax/tree/main/examples/seq2seq

## Requirements

`pip install -r requirements.txt`

## Run on cpu

`python train.py --num_train_steps 1 --decode_frequency 1`

## Run on spu
First, enable `"reveal_secret_indicies": true` in `SPU runtime_config`, 

the default configuration file locates at [examples/python/conf/3pc.json](../../conf/3pc.json).

Second, launch two terminals.

In the first terminal, start SPU backend:

`bazel run -c opt //examples/python/utils:nodectl -- up`

In the second terminal, run seq2seq example:

`bazel run -c opt //examples/python/ml/flax_seq2seq:train_on_spu -- --num_train_steps 1 --decode_frequency 1`
