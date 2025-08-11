"""
Enhanced HRM v2 with stability improvements and advanced features
- Gradient clipping and stable training
- Hybrid positional encodings (RoPE + ALiBi)
- Improved ACT mechanism with lower loss weighting
- Memory optimization with gradient checkpointing
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import torch.utils.checkpoint as checkpoint

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.text_embedding_v2 import TextEmbeddingV2


@dataclass
class HierarchicalReasoningModel_ACTV2InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV2Carry:
    inner_carry: HierarchicalReasoningModel_ACTV2InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV2Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"
    
    # Task type
    task: str = "puzzle"
    pad_token_id: int = 0
    
    # V2 Enhancements
    gradient_checkpointing: bool = True
    alibi_max_bias: float = 8.0  # ALiBi bias range
    act_loss_weight: float = 0.1  # Reduced from 0.5
    use_peri_ln: bool = True  # Peri-layer normalization


class HierarchicalReasoningModel_ACTV2Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV2Config) -> None:
        super().__init__()
        self.config = config

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=(config.task == "text_lm")
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps
        
        # Peri-LN: Additional normalization for stability
        if config.use_peri_ln:
            self.pre_attn_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.pre_mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.use_peri_ln:
            # Peri-LN: Pre-normalization + post-normalization
            residual = hidden_states
            hidden_states = self.pre_attn_norm(hidden_states)
            hidden_states = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = rms_norm(residual + hidden_states, variance_epsilon=self.norm_eps)
            
            residual = hidden_states
            hidden_states = self.pre_mlp_norm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = rms_norm(residual + hidden_states, variance_epsilon=self.norm_eps)
        else:
            # Original Post-Norm
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
            hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        
        return hidden_states


class HierarchicalReasoningModel_ACTV2ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV2Block], use_checkpointing: bool = True):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.use_checkpointing = use_checkpointing

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        
        # Layers with optional gradient checkpointing
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                hidden_states = checkpoint.checkpoint(layer, kwargs.get('cos_sin'), hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class ALiBiPositionalBias(nn.Module):
    """Attention with Linear Biases (ALiBi) for better length extrapolation"""
    def __init__(self, num_heads: int, max_bias: float = 8.0):
        super().__init__()
        self.num_heads = num_heads
        self.max_bias = max_bias
        
        # Create bias slopes for each head
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        
        slopes = torch.tensor(get_slopes(num_heads)).unsqueeze(1).unsqueeze(1)
        self.register_buffer('slopes', slopes)

    def forward(self, seq_len: int) -> torch.Tensor:
        # Create position matrix
        positions = torch.arange(seq_len, device=self.slopes.device).unsqueeze(0) - torch.arange(seq_len, device=self.slopes.device).unsqueeze(1)
        positions = positions.abs() * -1  # Make negative distances
        
        # Apply slopes to create bias matrix
        alibi_bias = positions.unsqueeze(0) * self.slopes  # [num_heads, seq_len, seq_len]
        return alibi_bias.clamp(min=-self.max_bias)


class HierarchicalReasoningModel_ACTV2_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV2Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Determine if this is text LM mode
        self.is_text_lm = getattr(config, "task", "puzzle") == "text_lm"
        
        if self.is_text_lm:
            self.input_embed = TextEmbeddingV2(
                vocab_size=self.config.vocab_size,
                d_model=self.config.hidden_size,
                max_len=self.config.seq_len,
                pad_token_id=getattr(self.config, "pad_token_id", 0),
                pos_encodings=self.config.pos_encodings,
            )
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        else:
            self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
            
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0 and not self.is_text_lm:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                  batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
        elif self.is_text_lm:
            self.puzzle_emb_len = 0

        # Enhanced Positional Encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                            max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                            base=self.config.rope_theta)
        elif self.config.pos_encodings == "hybrid":
            # Hybrid: RoPE + ALiBi
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                            max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                            base=self.config.rope_theta)
            self.alibi = ALiBiPositionalBias(self.config.num_heads, self.config.alibi_max_bias)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError(f"Unsupported positional encoding: {self.config.pos_encodings}")

        # Reasoning Layers with gradient checkpointing
        self.H_level = HierarchicalReasoningModel_ACTV2ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV2Block(self.config) for _ in range(self.config.H_layers)],
            use_checkpointing=self.config.gradient_checkpointing
        )
        self.L_level = HierarchicalReasoningModel_ACTV2ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV2Block(self.config) for _ in range(self.config.L_layers)],
            use_checkpointing=self.config.gradient_checkpointing
        )
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Enhanced Q head initialization for stability
        with torch.no_grad():
            # Xavier init for better gradient flow
            nn.init.xavier_uniform_(self.q_head.weight, gain=0.1)
            self.q_head.bias.fill_(-3.0)  # Reduced from -5 for less aggressive halting

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        if self.is_text_lm:
            embedding = self.input_embed(input.to(torch.int32))
            return self.embed_scale * embedding
        else:
            # Original puzzle embedding logic
            embedding = self.embed_tokens(input.to(torch.int32))

            if self.config.puzzle_emb_ndim > 0 and hasattr(self, 'puzzle_emb'):
                puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
                
                pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
                if pad_count > 0:
                    puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

                embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

            if self.config.pos_encodings == "learned":
                embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

            return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV2InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV2InnerCarry):
        # Ensure init buffers are on the same device as the carry
        device = carry.z_H.device
        h_init = self.H_init.to(device)
        l_init = self.L_init.to(device)
        
        return HierarchicalReasoningModel_ACTV2InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), h_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), l_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV2InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV2InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Add ALiBi bias if using hybrid positional encoding
        if self.config.pos_encodings == "hybrid" and hasattr(self, "alibi"):
            seq_info["alibi_bias"] = self.alibi(self.config.seq_len)

        # Input encoding  
        input_key = "input_ids" if "input_ids" in batch else "inputs"
        puzzle_ids = batch.get("puzzle_identifiers", torch.zeros_like(batch[input_key][:, :1]))
        input_embeddings = self._input_embeddings(batch[input_key], puzzle_ids)

        # Forward iterations with gradient checkpointing
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad with checkpointing
        if self.config.gradient_checkpointing and self.training:
            z_L = checkpoint.checkpoint(self.L_level, z_L, z_H + input_embeddings, seq_info, use_reentrant=False)
            z_H = checkpoint.checkpoint(self.H_level, z_H, z_L, seq_info, use_reentrant=False)
        else:
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV2InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        if self.is_text_lm:
            output = self.lm_head(z_H)
        else:
            output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head with improved stability
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV2(nn.Module):
    """Enhanced ACT wrapper with stability improvements."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV2Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV2_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        input_key = "input_ids" if "input_ids" in batch else "inputs"
        batch_size = batch[input_key].shape[0]

        return HierarchicalReasoningModel_ACTV2Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV2Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV2Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # Enhanced ACT with stability improvements
            if self.training and (self.config.halt_max_steps > 1):
                # More conservative halting decision
                halt_threshold = torch.tanh(q_halt_logits - q_continue_logits)  # Bounded decision
                halted = halted | (halt_threshold > 0.1)  # Higher threshold for stability

                # Reduced exploration for stability
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob * 0.5) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                # Stable target Q computation
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                target_q = torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits))
                outputs["target_q_continue"] = torch.sigmoid(target_q).clamp(0.01, 0.99)  # Numerical stability

        return HierarchicalReasoningModel_ACTV2Carry(new_inner_carry, new_steps, halted, new_current_data), outputs