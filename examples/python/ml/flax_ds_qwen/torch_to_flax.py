import flax.serialization
import jax
import jax.numpy as jnp
import torch
from flax.core.frozen_dict import freeze, unfreeze
from transformers import AutoModelForCausalLM

from examples.python.ml.flax_ds_qwen.model_flax import Qwen2Config, Qwen2ForCausalLM


def convert_pytorch_to_jax_cpu(pytorch_state_dict, flax_params):
    """
    Convert PyTorch state dict to JAX/Flax parameters using CPU memory.

    Args:
        pytorch_state_dict: PyTorch state dict containing model weights
        flax_params: Initial Flax parameter structure

    Returns:
        Converted Flax parameters
    """
    jax_params = unfreeze(flax_params)

    # Helper function to convert torch tensor to numpy array
    def torch_to_numpy(tensor):
        return tensor.cpu().numpy()

    # Same parameter mappings as before, but without "params" prefix since it's already in flax_params
    name_mapping = {
        "model.embed_tokens.weight": ("model", "embed_tokens", "embedding"),
        "model.norm.weight": ("model", "norm", "weight"),
        "lm_head.weight": ("lm_head", "kernel"),
    }

    layer_mapping = {
        "model.layers.{}.input_layernorm.weight": (
            "model",
            "layers_{}",
            "input_layernorm",
            "weight",
        ),
        "model.layers.{}.post_attention_layernorm.weight": (
            "model",
            "layers_{}",
            "post_attention_layernorm",
            "weight",
        ),
        "model.layers.{}.self_attn.q_proj.weight": (
            "model",
            "layers_{}",
            "self_attn",
            "q_proj",
            "kernel",
        ),
        "model.layers.{}.self_attn.q_proj.bias": (
            "model",
            "layers_{}",
            "self_attn",
            "q_proj",
            "bias",
        ),
        "model.layers.{}.self_attn.k_proj.weight": (
            "model",
            "layers_{}",
            "self_attn",
            "k_proj",
            "kernel",
        ),
        "model.layers.{}.self_attn.k_proj.bias": (
            "model",
            "layers_{}",
            "self_attn",
            "k_proj",
            "bias",
        ),
        "model.layers.{}.self_attn.v_proj.weight": (
            "model",
            "layers_{}",
            "self_attn",
            "v_proj",
            "kernel",
        ),
        "model.layers.{}.self_attn.v_proj.bias": (
            "model",
            "layers_{}",
            "self_attn",
            "v_proj",
            "bias",
        ),
        "model.layers.{}.self_attn.o_proj.weight": (
            "model",
            "layers_{}",
            "self_attn",
            "o_proj",
            "kernel",
        ),
        "model.layers.{}.mlp.gate_proj.weight": (
            "model",
            "layers_{}",
            "mlp",
            "gate_proj",
            "kernel",
        ),
        "model.layers.{}.mlp.up_proj.weight": (
            "model",
            "layers_{}",
            "mlp",
            "up_proj",
            "kernel",
        ),
        "model.layers.{}.mlp.down_proj.weight": (
            "model",
            "layers_{}",
            "mlp",
            "down_proj",
            "kernel",
        ),
    }

    # Process tensors in batches to save memory
    def process_tensor(tensor, needs_transpose=False):
        # Convert to NumPy first
        np_array = torch_to_numpy(tensor)
        if needs_transpose:
            np_array = np_array.T
        # Only convert to JAX array at the end
        return jnp.array(np_array)

    # Convert non-layer parameters
    for torch_name, jax_path in name_mapping.items():
        if torch_name in pytorch_state_dict:
            tensor = pytorch_state_dict[torch_name]
            needs_transpose = torch_name == "lm_head.weight"

            # Process tensor on CPU
            jax_tensor = process_tensor(tensor, needs_transpose)

            # Navigate and assign
            current_dict = jax_params
            for key in jax_path[:-1]:
                current_dict = current_dict[key]
            current_dict[jax_path[-1]] = jax_tensor

            # Clear PyTorch cache
            del tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Convert layer parameters
    num_layers = 28
    for layer_idx in range(num_layers):
        for torch_template, jax_path in layer_mapping.items():
            torch_name = torch_template.format(layer_idx)
            if torch_name in pytorch_state_dict:
                tensor = pytorch_state_dict[torch_name]
                needs_transpose = "proj.weight" in torch_name

                # Process tensor on CPU
                jax_tensor = process_tensor(tensor, needs_transpose)

                # Format and navigate path
                formatted_path = [
                    p.format(layer_idx) if "{}" in p else p for p in jax_path
                ]
                current_dict = jax_params
                for key in formatted_path[:-1]:
                    current_dict = current_dict[key]
                current_dict[formatted_path[-1]] = jax_tensor

                # Clear memory
                del tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return freeze(jax_params)


# Usage example:
def convert_model_cpu(flax_model, torch_model):
    # Initialize with minimal input on CPU
    key = jax.random.PRNGKey(0)
    init_inputs = jnp.ones((1, 1), dtype=jnp.int32)

    variables = flax_model.init(key, input_ids=init_inputs)

    # Load PyTorch weights
    pytorch_state_dict = torch_model.state_dict()

    # Convert weights on CPU
    converted_params = convert_pytorch_to_jax_cpu(
        pytorch_state_dict, variables["params"]
    )

    return converted_params


def torch_to_flax():
    # Load the PyTorch model
    torch_model = AutoModelForCausalLM.from_pretrained(
        "/data/models/DeepSeek-R1-Distill-Qwen-1.5B",
        # device_map="cpu",
        local_files_only=True,
    )

    # Load the Flax model
    with jax.default_device(jax.devices("cpu")[0]):
        flax_model = Qwen2ForCausalLM(Qwen2Config())

        # Convert the PyTorch parameters
        converted_params = convert_model_cpu(flax_model, torch_model)

        # Saving parameters:
        with open("flax_params.msgpack", "wb") as f:
            bytes_output = flax.serialization.to_bytes(converted_params)
            f.write(bytes_output)

    del torch_model
    del flax_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    del converted_params
    del bytes_output


if __name__ == "__main__":
    torch_to_flax()
