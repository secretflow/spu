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
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/ml/flax_gpt2/3pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_gpt2:flax_gpt2

import argparse
import json
import pdb
import sys
import os
from time import perf_counter
import torch
import jax.numpy as jnp
from transformers import BertTokenizer, BertConfig
from transformers import set_seed
import spu.spu_pb2 as spu_pb2

import spu.utils.distributed as ppd

if os.path.exists("/home/zyli/sparse_ppml2"):
    user_name="zyli"
    print(f"Using user_name: {user_name}")
    sys.path.append('/home/zyli/sparse_ppml2')
elif os.path.exists("/home/lizhengyi.lzy/sparse_ppml2"):
    user_name="lizhengyi.lzy"
    print(f"Using user_name: {user_name}")
    sys.path.append('/home/lizhengyi.lzy/sparse_ppml2')
else:
    raise("error")
from GLUE.model.utils import set_attention_sparsity


USE_SPARSE_TRANSFORMER=True
if USE_SPARSE_TRANSFORMER:
    from GLUE.model.modeling_flax_bert import FlaxBertForSequenceClassification
    from GLUE.model.modeling_bert import BertForSequenceClassification
else:
    from transformers import FlaxBertForSequenceClassification
    from transformers import BertForSequenceClassification

set_seed(42)

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_bert/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

copts = spu_pb2.CompilerOptions()

# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

# dump出compile pass的中间结果
copts.enable_pretty_print = True
copts.pretty_print_dump_dir = "ppdump"





def torch_baseline(inputs):
    model = BertForSequenceClassification.from_pretrained(model_name)
    # try:
    #     set_attention_sparsity(model, 'bert-base-uncased', 'clip_poly', -5, 5)
    # except:
    #     print("No attention sparsity")

    # Tokenize the input text
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits

def glue(input_ids, params):
    model = FlaxBertForSequenceClassification(config=config)

    start=perf_counter()
    outputs = model(input_ids=input_ids, params=params)
    print(f"Elapsed time: {perf_counter()-start}")
    return outputs.logits


def run_on_cpu(inputs):
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        inputs, return_tensors='jax', padding=True, truncation=True
    )
    outputs = glue(inputs_ids, pretrained_model.params)
    return outputs


def run_on_spu(inputs):
    # encode context the generation is conditioned on
    inputs_ids = tokenizer.encode(
        inputs, return_tensors='jax', padding=True, truncation=True
    )

    input_ids = ppd.device("P1")(lambda x: x)(inputs_ids)
    params = ppd.device("P2")(lambda x: x)(pretrained_model.params)
    outputs = ppd.device("SPU")(
        glue, copts=copts
    )(input_ids, params)
    outputs = ppd.get(outputs)
    return outputs


if __name__ == '__main__':
    NUM_HIDDEN_LAYERS=1

    ATTN_STRATEGY='clip_poly'
    CLIP_THRES=-4.0
    GELU_STRATEGY='clip_poly'
    GELU_ORDER=2

    sparse_pattern="8x8"
    task_name="stsb"
    sparsity=0.85
    model_name = f"/data/models/saved_models/pruned/bert-base-uncased/{task_name}/{sparse_pattern}_{sparsity}_20"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_hidden_layers=NUM_HIDDEN_LAYERS

    config.attn_strategy=ATTN_STRATEGY
    config.clip_thres=CLIP_THRES
    config.gelu_strategy=GELU_STRATEGY
    config.gelu_order=GELU_ORDER
    pretrained_model = FlaxBertForSequenceClassification.from_pretrained(model_name, config=config, from_pt=True,)

    # # seq_length=toy
    # inputs='I enjoy walking with my cute dog'

    # # seq_length=64
    # inputs="Lily woke up early on Saturday morning feeling excited. It was her 10th birthday and she couldn't wait to celebrate with her family and friends. She bounced out of the bed and put on her favorite dress that her grandma had given her. Lily rushed into the kitchen where her"

    # seq_length=128
    inputs="Lily woke up early on Saturday morning feeling excited. It was her 10th birthday and she couldn't wait to celebrate with her family and friends. She bounced out of the bed and put on her favorite dress that her grandma had given her. Lily rushed into the kitchen where her mom was making pancakes, her favorite breakfast. 'Morning sweetie! Happy birthday!' her mom said with a warm smile. Lily grinned and gave her mom a big hug. She helped set the table and her dad came downstairs holding presents. After eating yummy pancakes, Lily tore open her gifts finding art supplies, books, and a new bike helmet."

    # # seq_length=196
    # inputs="Lily woke up early on Saturday morning feeling excited. It was her 10th birthday and she couldn't wait to celebrate with her family and friends. She bounced out of the bed and put on her favorite dress that her grandma had given her. Lily rushed into the kitchen where her mom was making pancakes, her favorite breakfast. 'Morning sweetie! Happy birthday!' her mom said with a warm smile. Lily grinned and gave her mom a big hug. She helped set the table and her dad came downstairs holding presents. After eating yummy pancakes, Lily tore open her gifts finding art supplies, books, and a new bike helmet. Lily woke up early on Saturday morning feeling excited. It was her 10th birthday and she couldn't wait to celebrate with her family and friends. She bounced out of the bed and put on her favorite dress that her grandma had given her. Lily rushed into the kitchen where her mom was making pancakes. Then she went to the park with friends."

    # # seq_length=256
    # inputs='It was a beautiful summer day and Jeff decided to take a walk through the park. The sun was shining brightly and there was not a cloud in the sky. As he strolled along the path, he breathed in the fresh air and took in the lovely scenery around him. Several people were out enjoying the weather - some were walking dogs, others were having picnics, and children were playing on the playground equipment. Jeff smiled as he watched a group of kids laughing and chasing each other around. He continued following the tree-lined trail, listening to the chirping birds above. After winding through the lush green landscape for a while, he came upon a large pond. Ducks and geese floated gracefully across the still water. Jeff found an empty bench near the water edge and sat down to relax for a bit. He took out his phone and snapped some photos of the idyllic setting. As he sat there, he noticed a young couple paddling in a small boat in the middle of the pond. Jeff realized this park always lifted his spirits and he made a mental note to come here more often when he needed a peaceful retreat from the stresses of everyday life. After taking in the tranquil scene for a bit longer, Jeff got up and continued on his walk.'

    # # seq_length=512
    # inputs='It was a beautiful summer day and Jeff decided to take a walk through the park. The sun was shining brightly and there was not a cloud in the sky. As he strolled along the path, he breathed in the fresh air and took in the lovely scenery around him. Several people were out enjoying the weather - some were walking dogs, others were having picnics, and children were playing on the playground equipment. Jeff smiled as he watched a group of kids laughing and chasing each other around. He continued following the tree-lined trail, listening to the chirping birds above. After winding through the lush green landscape for a while, he came upon a large pond. Ducks and geese floated gracefully across the still water. Jeff found an empty bench near the water edge and sat down to relax for a bit. He took out his phone and snapped some photos of the idyllic setting. As he sat there, he noticed a young couple paddling in a small boat in the middle of the pond. Jeff realized this park always lifted his spirits and he made a mental note to come here more often when he needed a peaceful retreat from the stresses of everyday life. After taking in the tranquil scene for a bit longer, Jeff got up and continued on his walk. It was a beautiful summer day and Jeff decided to take a walk through the park. The sun was shining brightly and there was not a cloud in the sky. As he strolled along the path, he breathed in the fresh air and took in the lovely scenery around him. Several people were out enjoying the weather - some were walking dogs, others were having picnics, and children were playing on the playground equipment. Jeff smiled as he watched a group of kids laughing and chasing each other around. He continued following the tree-lined trail, listening to the chirping birds above. After winding through the lush green landscape for a while, he came upon a large pond. Ducks and geese floated gracefully across the still water. Jeff found an empty bench near the water edge and sat down to relax for a bit. He took out his phone and snapped some photos of the idyllic setting. As he sat there, he noticed a young couple paddling in a small boat in the middle of the pond. Jeff realized this park always lifted his spirits and he made a mental note to come here more often when he needed a peaceful retreat from the stresses of everyday life. After taking in the tranquil scene for a bit longer, Jeff got up and continued on his walk.'

    # # seq_length=1024，注意这个输入会被截断成512，因为bert最长输入就512。
    # inputs='It was a beautiful summer day and Jeff decided to take a walk through the park. The sun was shining brightly and there was not a cloud in the sky. As he strolled along the path, he breathed in the fresh air and took in the lovely scenery around him. Several people were out enjoying the weather - some were walking dogs, others were having picnics, and children were playing on the playground equipment. Jeff smiled as he watched a group of kids laughing and chasing each other around. He continued following the tree-lined trail, listening to the chirping birds above. After winding through the lush green landscape for a while, he came upon a large pond. Ducks and geese floated gracefully across the still water. Jeff found an empty bench near the water edge and sat down to relax for a bit. He took out his phone and snapped some photos of the idyllic setting. As he sat there, he noticed a young couple paddling in a small boat in the middle of the pond. Jeff realized this park always lifted his spirits and he made a mental note to come here more often when he needed a peaceful retreat from the stresses of everyday life. After taking in the tranquil scene for a bit longer, Jeff got up and continued on his walk. It was a beautiful summer day and Jeff decided to take a walk through the park. The sun was shining brightly and there was not a cloud in the sky. As he strolled along the path, he breathed in the fresh air and took in the lovely scenery around him. Several people were out enjoying the weather - some were walking dogs, others were having picnics, and children were playing on the playground equipment. Jeff smiled as he watched a group of kids laughing and chasing each other around. He continued following the tree-lined trail, listening to the chirping birds above. After winding through the lush green landscape for a while, he came upon a large pond. Ducks and geese floated gracefully across the still water. Jeff found an empty bench near the water edge and sat down to relax for a bit. He took out his phone and snapped some photos of the idyllic setting. As he sat there, he noticed a young couple paddling in a small boat in the middle of the pond. Jeff realized this park always lifted his spirits and he made a mental note to come here more often when he needed a peaceful retreat from the stresses of everyday life. After taking in the tranquil scene for a bit longer, Jeff got up and continued on his walk. It was a beautiful summer day and Jeff decided to take a walk through the park. The sun was shining brightly and there was not a cloud in the sky. As he strolled along the path, he breathed in the fresh air and took in the lovely scenery around him. Several people were out enjoying the weather - some were walking dogs, others were having picnics, and children were playing on the playground equipment. Jeff smiled as he watched a group of kids laughing and chasing each other around. He continued following the tree-lined trail, listening to the chirping birds above. After winding through the lush green landscape for a while, he came upon a large pond. Ducks and geese floated gracefully across the still water. Jeff found an empty bench near the water edge and sat down to relax for a bit. He took out his phone and snapped some photos of the idyllic setting. As he sat there, he noticed a young couple paddling in a small boat in the middle of the pond. Jeff realized this park always lifted his spirits and he made a mental note to come here more often when he needed a peaceful retreat from the stresses of everyday life. After taking in the tranquil scene for a bit longer, Jeff got up and continued on his walk. It was a beautiful summer day and Jeff decided to take a walk through the park. The sun was shining brightly and there was not a cloud in the sky. As he strolled along the path, he breathed in the fresh air and took in the lovely scenery around him. Several people were out enjoying the weather - some were walking dogs, others were having picnics, and children were playing on the playground equipment. Jeff smiled as he watched a group of kids laughing and chasing each other around. He continued following the tree-lined trail, listening to the chirping birds above. After winding through the lush green landscape for a while, he came upon a large pond. Ducks and geese floated gracefully across the still water. Jeff found an empty bench near the water edge and sat down to relax for a bit. He took out his phone and snapped some photos of the idyllic setting. As he sat there, he noticed a young couple paddling in a small boat in the middle of the pond. Jeff realized this park always lifted his spirits and he made a mental note to come here more often when he needed a peaceful retreat from the stresses of everyday life. After taking in the tranquil scene for a bit longer, Jeff got up and continued on his walk.'

    # print('\n------\nRun on Torch')
    # outputs_ids = torch_baseline(inputs)
    # print(outputs_ids)

    print('\n------\nRun on CPU')
    outputs_ids = run_on_cpu(inputs)
    print(outputs_ids)

    print('\n------\nRun on SPU')
    start=perf_counter()
    outputs_ids = run_on_spu(inputs)
    print(f"Elapsed time: {perf_counter()-start}")
    print(outputs_ids)
