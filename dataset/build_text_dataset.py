# dataset/build_text_dataset.py
import argparse, os, sys, json, glob, numpy as np
from typing import List
from transformers import AutoTokenizer

def _expand_inputs(items: List[str]) -> List[str]:
    out = []
    for s in items:
        if any(ch in s for ch in ["*", "?", "["]):
            out.extend(glob.glob(s))
        elif os.path.isdir(s):
            out.extend(glob.glob(os.path.join(s, "*.txt")))
        else:
            out.append(s)
    # unique + stable order
    return sorted(list(dict.fromkeys(out)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_glob", nargs="+", required=True,
                   help='One or more globs/paths. Example: "data/raw_text/*.txt" or a folder')
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer_name", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--shuffle", action="store_true", help="Shuffle tokens before packing")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = _expand_inputs(args.input_glob)
    if not files:
        print("No input .txt files found. Check your --input_glob.", file=sys.stderr)
        sys.exit(2)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Concatenate all text with separators to avoid accidental long words across joins
    corpus_chunks = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            corpus_chunks.append(f.read().strip())
    corpus = "\n\n".join(corpus_chunks)

    ids = tok(
        corpus,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None
    ).input_ids

    arr = np.array(ids, dtype=np.int32)
    if args.shuffle:
        # Optional: shuffle *tokens* (usually worse for LM; off by default)
        rng = np.random.default_rng(123)
        rng.shuffle(arr)

    # Pack into fixed blocks for causal LM
    n = (len(arr) // args.block_size) * args.block_size
    arr = arr[:n].reshape(-1, args.block_size)

    # Split
    n_val = max(1, int(len(arr) * args.val_ratio))
    val = arr[-n_val:]
    train = arr[:-n_val] if n_val < len(arr) else arr[:0]

    np.save(os.path.join(args.output_dir, "train.npy"), train)
    np.save(os.path.join(args.output_dir, "val.npy"), val)

    meta = {
        "tokenizer_name": args.tokenizer_name,
        "vocab_size": int(tok.vocab_size),
        "block_size": int(args.block_size),
        "pad_token_id": int(tok.pad_token_id),
        "eos_token_id": int(tok.eos_token_id),
        "num_train_blocks": int(train.shape[0]),
        "num_val_blocks": int(val.shape[0]),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {train.shape[0]} train blocks and {val.shape[0]} val blocks to {args.output_dir}")

if __name__ == "__main__":
    main()