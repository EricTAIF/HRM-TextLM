from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.text_embedding import TextEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
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


def build_causal_mask(T: int, device):
    return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=(config.task == "text_lm")  # Enable causal attention for text LM
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Determine if this is text LM mode
        self.is_text_lm = getattr(config, "task", "puzzle") == "text_lm"
        
        if self.is_text_lm:
            self.input_embed = TextEmbedding(
                vocab_size=self.config.vocab_size,
                d_model=self.config.hidden_size,
                max_len=self.config.seq_len,
                pad_token_id=getattr(self.config, "pad_token_id", 0),
                pos_encodings=self.config.pos_encodings,
            )
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        else:
            self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
            self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
            
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0 and not self.is_text_lm:
            # Zero init puzzle embeddings (only for puzzle tasks)
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
        elif self.is_text_lm:
            self.puzzle_emb_len = 0  # No puzzle embeddings for text

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        if self.is_text_lm:
            # For text LM, input contains input_ids directly
            embedding = self.input_embed(input.to(torch.int32))  # [B, T, H]
            # Adjust time dimension to model seq_len by padding/truncating
            B, T, H = embedding.shape
            if T < self.config.seq_len:
                pad_T = self.config.seq_len - T
                embedding = torch.nn.functional.pad(embedding, (0, 0, 0, pad_T))  # pad time dim to the right
            elif T > self.config.seq_len:
                embedding = embedding[:, -self.config.seq_len:, :]
            # Ensure forward dtype (fp16/bf16) for FlashAttention compatibility
            return self.embed_scale * embedding.to(self.forward_dtype)
        else:
            # Original puzzle embedding logic
            # Token embedding
            embedding = self.embed_tokens(input.to(torch.int32))

            # Puzzle embeddings
            if self.config.puzzle_emb_ndim > 0 and hasattr(self, 'puzzle_emb'):
                puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
                
                pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
                if pad_count > 0:
                    puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

                embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

            # Position embeddings
            if self.config.pos_encodings == "learned":
                # scale by 1/sqrt(2) to maintain forward variance
                embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

            # Scale
            return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        device = self.H_init.device
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor], return_trace: bool = False):
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Causal attention is handled in the attention layer initialization

        # Input encoding  
        input_key = "input_ids" if "input_ids" in batch else "inputs"
        puzzle_ids = batch.get("puzzle_identifiers", torch.zeros_like(batch[input_key][:, :1]))
        input_embeddings = self._input_embeddings(batch[input_key], puzzle_ids)

        # Forward iterations
        trace = [] if return_trace else None
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    last_combo = (_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)
                    if not last_combo:
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                        if return_trace:
                            # record per-token norm for z_L
                            trace.append({
                                "level": "L",
                                "h": _H_step,
                                "l": _L_step,
                                "z_norm": z_L.norm(dim=-1).to(torch.float32).cpu(),  # [B, T]
                            })

                if _H_step != self.config.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, **seq_info)
                    if return_trace:
                        trace.append({
                            "level": "H",
                            "h": _H_step,
                            "l": None,
                            "z_norm": z_H.norm(dim=-1).to(torch.float32).cpu(),
                        })

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        
        if self.is_text_lm:
            output = self.lm_head(z_H)  # For text LM, return full sequence logits
        else:
            output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return (new_carry, output, (q_logits[..., 0], q_logits[..., 1]), trace)


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        # Handle both "inputs" (puzzle) and "input_ids" (text) keys
        input_key = "input_ids" if "input_ids" in batch else "inputs"
        input_tensor = batch[input_key]
        batch_size = input_tensor.shape[0]
        device = input_tensor.device

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # allocated on model device
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            # Only include tensor entries from batch for current_data
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        return_trace = bool(batch.get("debug_trace", False))
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), trace = self.inner(new_inner_carry, new_current_data, return_trace=return_trace)
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data, return_trace=False)[-1]
                
                outputs_target_q_continue = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        # Package outputs (after computing steps/halted)
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            # Debug/visualization-friendly extras (use current carry tensors)
            "z_H": new_inner_carry.z_H,
            "z_L": new_inner_carry.z_L,
            "steps": new_steps,
            "halted": halted,
        }
        # Re-run inner with tracing if requested via outer API through loss head
        # We detect this in losses head; here we only include trace if already computed above.
        if trace is not None:
            outputs["trace"] = trace
        if self.training and (self.config.halt_max_steps > 1):
            outputs["target_q_continue"] = outputs_target_q_continue  # type: ignore[name-defined]

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
