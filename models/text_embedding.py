# models/text_embedding.py
import torch
import torch.nn as nn

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, pad_token_id: int, pos_encodings: str = "learned"):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.use_learned_pos = (pos_encodings == "learned")
        if self.use_learned_pos:
            self.pos = nn.Embedding(max_len, d_model)
        else:
            self.pos = None  # if HRM core applies RoPE/RoPE-like inside attention

    def forward(self, input_ids: torch.LongTensor):
        # input_ids: [B, T]
        x = self.tok(input_ids)
        if self.pos is not None:
            T = x.size(1)
            pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
            x = x + self.pos(pos_ids)
        return x