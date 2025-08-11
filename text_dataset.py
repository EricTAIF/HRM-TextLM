# text_dataset.py
import os, json, numpy as np, torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100

class TextBlocks(Dataset):
    """
    Expects:
      data_dir/
        train.npy  [N, T] int32
        val.npy    [M, T] int32
        meta.json  {"block_size":..., "pad_token_id":..., "vocab_size":...}
    """
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        fp = os.path.join(data_dir, f"{split}.npy")
        self.X = np.load(fp, mmap_mode="r")  # memory-friendly

        self.block_size = int(self.meta["block_size"])
        self.pad_id = int(self.meta["pad_token_id"])
        self.vocab_size = int(self.meta["vocab_size"])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.int64))  # [T]
        # Next-token labels: shift left by 1
        y = torch.roll(x, shifts=-1)
        y[-1] = IGNORE_INDEX
        attn = torch.ones_like(x)
        return {"input_ids": x, "labels": y, "attention_mask": attn}