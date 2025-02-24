# adapted from https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B


from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax


class Qwen2Config:
    def __init__(
        self,
        vocab_size=151936,  # Vocabulary size for token embeddings.
        hidden_size=1536,  # Hidden size of the transformer.
        intermediate_size=8960,  # Size of the intermediate layer in the FFN.
        num_hidden_layers=28,  # Number of transformer (decoder) layers.
        num_attention_heads=12,  # Total number of attention heads.
        num_key_value_heads=2,  # Fewer key/value heads used for Grouped Query Attention (GQA).
        max_position_embeddings=131072,  # Maximum context length.
        initializer_range=0.02,
        rms_norm_eps=1e-6,  # Small epsilon for RMSNorm stability.
        rope_theta=10000.0,  # Parameter for Rotary Positional Embeddings (RoPE).
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout


class Qwen2RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Initialize the scaling weight (set to ones).
        self.weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.hidden_size,),
            self.param_dtype,
        )

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        # Compute variance over the last dimension.
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        # Normalize using the reciprocal square root.
        hidden_states = hidden_states * lax.rsqrt(variance + self.eps)
        # Scale and cast back to original dtype.
        return (self.weight * hidden_states).astype(input_dtype)


class Qwen2MLP(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Gate projection (part of SwiGLU; no bias)
        self.gate_proj = nn.Dense(
            features=self.config.intermediate_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        # Up projection (another part of SwiGLU; no bias)
        self.up_proj = nn.Dense(
            features=self.config.intermediate_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        # Down projection (to bring back to hidden size; no bias)
        self.down_proj = nn.Dense(
            features=self.config.hidden_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        # SwiGLU activation: apply SiLU on gate_proj(x), multiply elementwise with up_proj(x),
        # and then project down back to hidden size.
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    config: Qwen2Config
    layer_idx: int
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Compute per-head dimension.
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        # Calculate number of groups for GQA
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        # Scaling factor for dot-product attention.
        self.scaling = self.head_dim**-0.5

        # Query projection: projects to full number of query heads.
        self.q_proj = nn.Dense(
            features=self.config.num_attention_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        # Key projection: projects to a smaller number of key heads (GQA).
        self.k_proj = nn.Dense(
            features=self.config.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        # Value projection: similar to key projection.
        self.v_proj = nn.Dense(
            features=self.config.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        # Output projection: brings the concatenated heads back to hidden size.
        self.o_proj = nn.Dense(
            features=self.config.hidden_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_embeddings: Tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        # Extract batch size and sequence length.
        batch_size, seq_length = hidden_states.shape[:2]

        # Compute query, key, and value projections.
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose queries: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        query_states = query_states.reshape(
            batch_size, seq_length, self.config.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        # For keys and values, use the reduced number of key/value heads.
        key_states = key_states.reshape(
            batch_size, seq_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Unpack rotary positional embeddings (RoPE).
        cos, sin = position_embeddings
        # Apply RoPE to both query and key states.
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Repeat key and value states along the head dimension to match query heads (GQA).
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Compute scaled dot-product attention.
        attn_weights = (
            jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * self.scaling
        )

        # Incorporate the causal (and any provided) attention mask.
        attn_weights = attn_weights + attention_mask

        # Softmax to get attention probabilities.
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Optionally apply dropout during training.
        if not deterministic and self.config.attention_dropout > 0:
            attn_weights = nn.dropout(
                attn_weights,
                rate=self.config.attention_dropout,
                deterministic=deterministic,
            )

        # Compute attention output.
        attn_output = jnp.matmul(attn_weights, value_states)
        # Reshape back: transpose and merge head dimensions.
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, -1)

        # Final output projection.
        attn_output = self.o_proj(attn_output)

        # Return the attention output and optionally the attention weights.
        return attn_output, attn_weights if output_attentions else None

    @nn.nowrap
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input.
        Splits the last dimension in half and concatenates the negative of the second half with the first half.
        """
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    @nn.nowrap
    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        # Expand cos and sin for broadcasting.
        cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
        sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
        # Apply RoPE to queries and keys.
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    @nn.nowrap
    def repeat_kv(self, hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
        # If no repetition is needed, return as is.
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        # Add a new dimension for repetition.
        hidden_states = jnp.expand_dims(hidden_states, axis=2)
        # Broadcast along the new dimension.
        hidden_states = jnp.broadcast_to(
            hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim)
        )
        # Reshape to merge the repeated dimension with the key/value head dimension.
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2DecoderLayer(nn.Module):
    config: Qwen2Config
    layer_idx: int
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Self-attention block
        self.self_attn = Qwen2Attention(
            config=self.config,
            layer_idx=self.layer_idx,
            param_dtype=self.param_dtype,
        )
        # Feed-forward block (with SwiGLU activation)
        self.mlp = Qwen2MLP(
            config=self.config,
            param_dtype=self.param_dtype,
        )
        # Pre-attention normalization (RMSNorm)
        self.input_layernorm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )
        # Post-attention normalization (RMSNorm)
        self.post_attention_layernorm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        # Save the input for the first residual connection.
        residual = hidden_states
        # Pre-normalization before self-attention.
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention computation with GQA, RoPE, and QKV bias.
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )
        # Residual connection after attention.
        hidden_states = residual + hidden_states

        # Fully connected MLP block:
        residual = hidden_states
        # Pre-normalization before the MLP.
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Apply the MLP with SwiGLU activation.
        hidden_states = self.mlp(hidden_states)
        # Residual connection after the MLP.
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2Model(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Token embedding layer.
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

        # Create a list of decoder layers.
        self.layers = [
            Qwen2DecoderLayer(
                config=self.config,
                layer_idx=i,
                param_dtype=self.param_dtype,
            )
            for i in range(self.config.num_hidden_layers)
        ]

        # Final RMSNorm layer.
        self.norm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # If input embeddings aren’t provided, compute them from input_ids.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Generate position IDs if not provided.
        if position_ids is None:
            position_ids = jnp.arange(inputs_embeds.shape[1])[None, :]

        # Create a causal attention mask to enforce autoregressive behavior.
        causal_mask = self._prepare_causal_attention_mask(
            attention_mask,
            inputs_embeds.shape[1],
            inputs_embeds.dtype,
            inputs_embeds.shape[0],
        )

        hidden_states = inputs_embeds
        # Compute rotary positional embeddings (RoPE) for use in each attention layer.
        position_embeddings = self._compute_rope_embeddings(hidden_states, position_ids)

        # Optionally collect all hidden states and attentions.
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Pass through each decoder layer.
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # Final layer normalization.
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_attentions,)

        return outputs

    @nn.nowrap
    def _compute_rope_embeddings(self, hidden_states, position_ids, inv_freq=None):
        # Compute rotary positional embeddings (RoPE).
        if inv_freq is None:
            dim = self.config.hidden_size // self.config.num_attention_heads
            # Compute inverse frequencies based on rope_theta.
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
            )

        # Adjust dimensions for broadcasting.
        inv_freq = jnp.expand_dims(jnp.expand_dims(inv_freq, 0), 0)
        position_ids = jnp.expand_dims(position_ids.astype(jnp.float32), -1)
        freqs = jnp.matmul(position_ids, inv_freq)
        # Duplicate the frequencies to form cosine and sine inputs.
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return cos, sin

    @nn.nowrap
    def _prepare_causal_attention_mask(
        self,
        attention_mask: Optional[jnp.ndarray],
        sequence_length: int,
        dtype: jnp.dtype,
        batch_size: int,
    ):
        # Create a causal mask using an upper triangular matrix.
        min_dtype = jnp.finfo(dtype).min
        causal_mask = jnp.triu(
            jnp.full((sequence_length, sequence_length), min_dtype, dtype=dtype),
            k=1,
        )
        # Broadcast to shape (batch, 1, seq, seq).
        causal_mask = jnp.broadcast_to(
            causal_mask[None, None, :, :],
            (batch_size, 1, sequence_length, sequence_length),
        )

        # If an additional attention mask is provided, add it.
        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask

        return causal_mask

    @nn.nowrap
    def _compute_rope_embeddings(self, hidden_states, position_ids, inv_freq=None):
        if inv_freq is None:
            dim = self.config.hidden_size // self.config.num_attention_heads
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
            )

        inv_freq = jnp.expand_dims(jnp.expand_dims(inv_freq, 0), 0)
        position_ids = jnp.expand_dims(position_ids.astype(jnp.float32), -1)
        freqs = jnp.matmul(position_ids, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return cos, sin

    @nn.nowrap
    def _prepare_causal_attention_mask(
        self,
        attention_mask: Optional[jnp.ndarray],
        sequence_length: int,
        dtype: jnp.dtype,
        batch_size: int,
    ):
        min_dtype = jnp.finfo(dtype).min
        causal_mask = jnp.triu(
            jnp.full((sequence_length, sequence_length), min_dtype, dtype=dtype),
            k=1,
        )
        causal_mask = jnp.broadcast_to(
            causal_mask[None, None, :, :],
            (batch_size, 1, sequence_length, sequence_length),
        )

        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask

        return causal_mask


class Qwen2ForCausalLM(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Instantiate the core transformer model.
        self.model = Qwen2Model(
            config=self.config,
            param_dtype=self.param_dtype,
        )
        # Final LM head to project hidden states to vocabulary logits.
        self.lm_head = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, input_ids, *args, **kwargs):
        outputs = self.model(input_ids, *args, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return (logits,) + outputs[1:]

    def generate(
        self,
        params,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        prng_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Optimized generation using fixed-size padding and sharding.
        """

        # Determine the padded sequence length (input length + new tokens).
        max_sequence_length = input_ids.shape[1] + max_new_tokens
        batch_size = input_ids.shape[0]

        # Create padded input buffers.
        padded_input = jnp.pad(
            input_ids,
            ((0, 0), (0, max_sequence_length - input_ids.shape[1])),
            constant_values=0,
        )
        attention_mask = jnp.ones_like(input_ids)
        attention_mask = jnp.pad(
            attention_mask,
            ((0, 0), (0, max_sequence_length - attention_mask.shape[1])),
            constant_values=0,
        )

        def sharded_step(sharded_params, padded_ids, attn_mask):
            # Run the forward pass (without extra parameter nesting).
            outputs = self.apply(
                sharded_params,
                input_ids=padded_ids,
                attention_mask=attn_mask,
                deterministic=True,
            )
            logits = outputs[0]  # extract logits

            # Determine the current sequence length from the attention mask.
            curr_seq_len = jnp.sum(attn_mask, axis=-1)

            # Gather logits corresponding to the current (last) token.
            batch_indices = jnp.arange(logits.shape[0])
            next_token_logits = logits[batch_indices, curr_seq_len - 1]

            # Apply temperature scaling.
            next_token_logits = next_token_logits / temperature

            # Optionally apply top-k filtering.
            if top_k is not None:
                top_logits, _ = jax.lax.top_k(next_token_logits, top_k)
                k_threshold = jnp.min(top_logits, axis=-1, keepdims=True)
                next_token_logits = jnp.where(
                    next_token_logits < k_threshold, -jnp.inf, next_token_logits
                )

            return next_token_logits

        # Autoregressive generation loop.
        current_input_length = input_ids.shape[1]

        for i in range(max_new_tokens):
            # Compute next token logits using the sharded function.
            next_token_logits = sharded_step(params, padded_input, attention_mask)

            # Sample (if do_sample=True) or take the argmax.
            if do_sample and prng_key is not None:
                prng_key, subkey = jax.random.split(prng_key)
                next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
            else:
                next_token = jnp.argmax(next_token_logits, axis=-1)

            # Insert the predicted token into the padded input.
            next_token = next_token[:, None]
            current_input_length += 1

            padded_input = padded_input.at[:, current_input_length - 1].set(
                next_token[:, 0]
            )
            attention_mask = attention_mask.at[:, current_input_length - 1].set(1)

        # Return the generated sequence without the padded region.
        return padded_input[:, :current_input_length]
