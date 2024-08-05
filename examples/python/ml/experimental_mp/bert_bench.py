# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/experimental_mp/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/experimental_mp:bert_bench

# Reference: https://huggingface.co/docs/transformers/model_doc/roberta#transformers.FlaxRobertaForSequenceClassification
from transformers import AutoTokenizer, FlaxBertForSequenceClassification, BertConfig
from datasets import load_dataset
import jax.numpy as jnp
import argparse
import json
import spu.utils.distributed as ppd
import time
import jax

from hack_functions import hack_gelu_context, hack_softmax_context
import spu.spu_pb2 as spu_pb2


def parse_args():
    parser = argparse.ArgumentParser(description="distributed driver.")
    parser.add_argument(
        "--config", default="examples/python/ml/experimental_mp/3pc.json", type=str
    )
    parser.add_argument("--fp16_w", action="store_true")
    parser.add_argument("--gelu", default="raw", help="['raw', 'quad', 'poly']")
    parser.add_argument("--softmax", default="raw", help="['raw', '2relu', '2quad']")
    parser.add_argument("--model", default="Bert", type=str)
    return parser.parse_args()


args = parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

whether_hack_gelu = False if args.gelu == "raw" else True
whether_hack_softmax = False if args.softmax == "raw" else True

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = True
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

model_path = "/data/transformers/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
# tokenizer.pad_token = tokenizer.eos_token
pretrained_model = FlaxBertForSequenceClassification.from_pretrained(
    model_path, from_pt=True
)

# Converting model params to FP16
if args.fp16_w:
    print(f"==== Converting fp16 weights ====")
    from flax import traverse_util

    flat_params = traverse_util.flatten_dict(pretrained_model.params)
    # for path in flat_params:
    #     print(path)
    mask = {
        path: (
            path[-2:] != ("classifier", "bias")
            and path[-2:] != ("classifier", "kernel")  # should match to var name
            # and path[-1] != "embedding"
        )
        for path in flat_params
    }
    mask = traverse_util.unflatten_dict(mask)
    print(mask)
    pretrained_model.params = pretrained_model.to_fp16(pretrained_model.params, mask)


def run_on_cpu(input_ids, attention_masks, labels):
    print(f"Running on CPU ...")
    params = pretrained_model.params

    def eval(params, input_ids, attention_masks):
        # config = BertConfig()
        # model = FlaxBertForSequenceClassification(config=config)
        model = pretrained_model
        logits = model(input_ids, attention_masks, params=params, train=False).logits
        label = jnp.argmax(logits, axis=-1)
        return logits, label

    start = time.time()
    with hack_gelu_context(args.gelu, enabled=True), hack_softmax_context(
        args.softmax, enabled=whether_hack_softmax
    ):
        logits, label = eval(params, input_ids, attention_masks)
    end = time.time()
    print(f"CPU runtime: {(end - start)}s\noutput logits: {logits}, label: {label}")


def run_on_spu(input_ids, attention_masks, labels):
    print(f"Running on SPU ...")
    params = pretrained_model.params

    def eval(params, input_ids, attention_masks):
        # config = BertConfig()
        # model = FlaxBertForSequenceClassification(config=config)
        model = pretrained_model
        logits = model(input_ids, attention_masks, params=params, train=False).logits
        label = jnp.argmax(logits, axis=-1)
        return logits, label

    spu_input_ids = ppd.device("P1")(lambda x: x)(input_ids)
    spu_attention_masks = ppd.device("P1")(lambda x: x)(attention_masks)
    spu_params = ppd.device("P1")(lambda x: x)(params)
    start = time.time()
    with hack_gelu_context(args.gelu, enabled=True), hack_softmax_context(
        args.softmax, enabled=whether_hack_softmax
    ):
        logits_spu, label_spu = ppd.device("SPU")(eval, copts=copts)(
            spu_params, spu_input_ids, spu_attention_masks
        )
    end = time.time()

    print(
        f"SPU runtime: {(end - start)}s\noutput logits: {ppd.get(logits_spu)}, label: {ppd.get(label_spu)}"
    )


def main():
    dataset = load_dataset("glue", "cola", split="train")
    batch_size = 2
    max_seq_length = 128
    dummy_input = dataset[:batch_size]

    features, labels = dummy_input["sentence"], dummy_input["label"]
    print(type(features))

    input_ids, attention_masks = (
        tokenizer(
            features,
            return_tensors="jax",
            padding="max_length",
            max_length=max_seq_length,
        )["input_ids"],
        tokenizer(
            features,
            return_tensors="jax",
            padding="max_length",
            max_length=max_seq_length,
        )["attention_mask"],
    )

    print("Inference ...")

    run_on_cpu(input_ids, attention_masks, labels)
    run_on_spu(input_ids, attention_masks, labels)


if __name__ == "__main__":
    main()
