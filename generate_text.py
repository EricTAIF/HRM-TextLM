# generate_text.py
import argparse, os, yaml, torch
from transformers import AutoTokenizer
from utils.functions import load_model_class


def _build_model_from_checkpoint(step_path: str, tokenizer, task: str = "text_lm"):
    # Resolve checkpoint dir and step file
    step_path = os.path.abspath(step_path)
    ckpt_dir = os.path.dirname(step_path)

    # Load saved training config if present
    cfg_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Cannot find all_config.yaml next to checkpoint: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Dataset metadata for vocab/seq
    data_meta = None
    try:
        meta_fp = os.path.join(cfg["data_path"], "meta.json")
        import json
        with open(meta_fp, "r") as mf:
            data_meta = json.load(mf)
    except Exception:
        pass

    vocab_size = data_meta.get("vocab_size", tokenizer.vocab_size) if data_meta else tokenizer.vocab_size
    seq_len = data_meta.get("block_size", 1024) if data_meta else 1024
    pad_id = data_meta.get("pad_token_id", tokenizer.pad_token_id) if data_meta else tokenizer.pad_token_id

    # Build model + loss head from saved arch config
    arch = cfg["arch"]
    model_cls = load_model_class(arch["name"])  # e.g. models.hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1
    loss_head_cls = load_model_class(arch["loss"]["name"])  # e.g. losses@ACTLossHead

    # Copy extra arch params (hidden_size, heads, etc.)
    model_cfg = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    model_cfg.update({
        "batch_size": 1,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "num_puzzle_identifiers": 1,
        "task": task,
        "pad_token_id": pad_id,
    })
    # Prefer bf16 if available, else fp16
    try:
        model_cfg.setdefault("forward_dtype", "bfloat16" if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)() else "float16")
    except Exception:
        model_cfg.setdefault("forward_dtype", "float16")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        base_model = model_cls(model_cfg)
        model = loss_head_cls(base_model, **{k: v for k, v in arch["loss"].items() if k != "name"})
        model = model.to(device)

        # Load weights
        state = torch.load(step_path, map_location=device)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], torch.nn.Module):
            model.load_state_dict(state["model"].state_dict(), strict=False)
        elif isinstance(state, dict):
            model.load_state_dict(state, strict=False)
        else:
            raise ValueError("Unsupported checkpoint format: expected state_dict or dict with 'model'")

    model.eval()
    return model

@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=100, top_p=0.9, temperature=1.0, stop_id=None, greedy: bool=False):
    model.eval()
    device = next(model.parameters()).device
    x = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    for _ in range(max_new_tokens):
        # Create batch format expected by HRM model
        batch = {"input_ids": x, "attention_mask": torch.ones_like(x)}
        
        # Get initial carry
        carry = model.initial_carry(batch)
        
        # Forward pass
        try:
            # Loss-head wrapper path
            carry, _, _, outputs, _ = model(carry=carry, batch=batch, return_keys=["logits"])  # type: ignore
            logits = outputs["logits"]
        except TypeError:
            # Bare model path
            carry, outputs = model(carry=carry, batch=batch)  # type: ignore
            logits = outputs["logits"]  # [1, T, V]
        
        # Use logits at the last real token position (before padding)
        cur_len = x.size(1)
        try:
            max_ctx = getattr(getattr(model, "model", model), "inner").config.seq_len  # type: ignore[attr-defined]
        except Exception:
            max_ctx = cur_len
        last_idx = min(cur_len, max_ctx) - 1
        if greedy or temperature <= 0 or top_p <= 0:
            next_id = torch.argmax(logits[:, last_idx, :], dim=-1)
        else:
            logits = logits[:, last_idx, :] / max(1e-6, temperature)
            # nucleus sampling
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > top_p).float().argmax(dim=-1).item()
            keep_idx = sorted_idx[:, :cutoff+1]
            keep_probs = torch.gather(probs, 1, keep_idx)
            keep_probs = keep_probs / keep_probs.sum(dim=-1, keepdim=True)
            next_id = keep_idx[0, torch.multinomial(keep_probs[0], 1)]
        x = torch.cat([x, next_id.view(1, 1)], dim=1)
        if stop_id is not None and next_id.item() == stop_id:
            break
    return tok.decode(x[0].tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer_name", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = _build_model_from_checkpoint(args.checkpoint, tok, task="text_lm")

    txt = generate(
        model, tok, args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        stop_id=tok.eos_token_id,
        greedy=args.greedy,
    )
    print(txt)

if __name__ == "__main__":
    main()
