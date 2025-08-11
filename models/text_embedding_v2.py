"""
Enhanced Text Embedding V2 with hybrid positional encodings and stability improvements
"""

import torch
import torch.nn as nn
import math


class TextEmbeddingV2(nn.Module):
    """Enhanced text embedding with hybrid positional encoding support"""
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int, pad_token_id: int, pos_encodings: str = "learned"):
        super().__init__()
        self.d_model = d_model
        self.pos_encodings = pos_encodings
        
        # Token embedding with improved initialization
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        nn.init.xavier_uniform_(self.tok.weight, gain=1.0)
        # Zero out padding token
        with torch.no_grad():
            self.tok.weight[pad_token_id].fill_(0)
        
        # Positional encodings
        if pos_encodings == "learned":
            self.pos = nn.Embedding(max_len, d_model)
            nn.init.xavier_uniform_(self.pos.weight, gain=0.1)  # Smaller init for stability
        elif pos_encodings == "sinusoidal":
            self.register_buffer("pos_encoding", self._create_sinusoidal_encoding(max_len, d_model))
            self.pos = None
        elif pos_encodings in ["rope", "hybrid"]:
            self.pos = None  # Handled by HRM core
        else:
            raise ValueError(f"Unsupported positional encoding: {pos_encodings}")
        
        # Learnable scaling factor for better gradient flow
        self.embed_scale = nn.Parameter(torch.tensor(math.sqrt(d_model)))
        
        # Optional layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings"""
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] token indices
            
        Returns:
            embeddings: [B, T, d_model] embedded tokens with positional info
        """
        B, T = input_ids.shape
        
        # Token embeddings
        x = self.tok(input_ids)  # [B, T, d_model]
        
        # Add positional information
        if self.pos_encodings == "learned":
            pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos(pos_ids)
            x = x + pos_emb
        elif self.pos_encodings == "sinusoidal":
            x = x + self.pos_encoding[:, :T, :]
        # For "rope" and "hybrid", positional info is handled by attention layers
        
        # Apply scaling and normalization
        x = x * self.embed_scale
        x = self.layer_norm(x)
        
        return x


class EnhancedRotaryEmbedding(nn.Module):
    """Enhanced RoPE with better numerical stability and extrapolation"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, 
                 scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Create frequency matrix with improved numerical stability
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute for common sequence lengths
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values for efficiency"""
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return
            
        self._seq_len_cached = max(seq_len, self._seq_len_cached)
        
        # Apply scaling for extrapolation
        scaled_max_pos = self._seq_len_cached * self.scaling_factor
        t = torch.arange(scaled_max_pos, device=device, dtype=dtype) / self.scaling_factor
        
        freqs = torch.outer(t, self.inv_freq.to(device))
        
        # Use complex exponential for better numerical stability
        emb = torch.polar(torch.ones_like(freqs), freqs)
        
        self._cos_cached = emb.real.to(dtype)
        self._sin_cached = emb.imag.to(dtype)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32):
        """
        Returns:
            cos, sin: [seq_len, dim//2] rotation matrices
        """
        self._update_cache(seq_len, device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


class AdaptivePositionalEmbedding(nn.Module):
    """Adaptive positional embedding that combines multiple encoding schemes"""
    
    def __init__(self, d_model: int, max_len: int = 2048, num_schemes: int = 3):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.num_schemes = num_schemes
        
        # Multiple positional encoding schemes
        chunk_size = d_model // num_schemes
        remainder = d_model % num_schemes
        
        self.schemes = nn.ModuleList()
        
        # Learned embeddings
        self.schemes.append(nn.Embedding(max_len, chunk_size))
        
        # Sinusoidal with different frequencies
        if num_schemes > 1:
            self.register_buffer("sinusoidal_1", 
                               self._create_sinusoidal(max_len, chunk_size, base=10000))
        if num_schemes > 2:
            self.register_buffer("sinusoidal_2", 
                               self._create_sinusoidal(max_len, chunk_size + remainder, base=1000))
        
        # Learnable mixing weights
        self.mixing_weights = nn.Parameter(torch.ones(num_schemes) / num_schemes)
        
    def _create_sinusoidal(self, max_len: int, d_model: int, base: float = 10000.0):
        """Create sinusoidal encoding with specified base frequency"""
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(base) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            pos_emb: [seq_len, d_model] mixed positional embeddings
        """
        positions = torch.arange(seq_len, device=device)
        
        embeddings = []
        
        # Learned embeddings
        embeddings.append(self.schemes[0](positions))
        
        # Sinusoidal embeddings
        if self.num_schemes > 1:
            embeddings.append(self.sinusoidal_1[:seq_len].to(device))
        if self.num_schemes > 2:
            embeddings.append(self.sinusoidal_2[:seq_len].to(device))
        
        # Mix embeddings with learnable weights
        weights = torch.softmax(self.mixing_weights, dim=0)
        mixed_emb = sum(w * emb for w, emb in zip(weights, embeddings))
        
        return mixed_emb