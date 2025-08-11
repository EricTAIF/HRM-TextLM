import argparse, os, json, yaml, torch, math
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from utils.functions import load_model_class


def build_model(checkpoint: str, tokenizer):
    ckpt_path = os.path.abspath(checkpoint)
    ckpt_dir = os.path.dirname(ckpt_path)
    cfg_fp = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.isfile(cfg_fp):
        raise FileNotFoundError(f"Missing all_config.yaml beside checkpoint: {cfg_fp}")
    with open(cfg_fp, "r") as f:
        cfg = yaml.safe_load(f)

    # Load dataset meta for vocab/seq
    meta = {}
    try:
        with open(os.path.join(cfg["data_path"], "meta.json"), "r") as mf:
            meta = json.load(mf)
    except Exception:
        pass

    arch = cfg["arch"]
    model_cls = load_model_class(arch["name"])  # models.hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1
    loss_head_cls = load_model_class(arch["loss"]["name"])  # losses@ACTLossHead

    model_cfg = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    model_cfg.update({
        "batch_size": 1,
        "seq_len": meta.get("block_size", 1024),
        "vocab_size": meta.get("vocab_size", tokenizer.vocab_size),
        "num_puzzle_identifiers": 1,
        "task": "text_lm",
        "pad_token_id": meta.get("pad_token_id", tokenizer.pad_token_id),
        "forward_dtype": "bfloat16" if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)() else "float16",
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        core = model_cls(model_cfg)
        model = loss_head_cls(core, **{k: v for k, v in arch["loss"].items() if k != "name"})
        model = model.to(device)

        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and all(k in state for k in ("model",)) and isinstance(state["model"], torch.nn.Module):
            model.load_state_dict(state["model"].state_dict(), strict=False)
        elif isinstance(state, dict):
            model.load_state_dict(state, strict=False)
        else:
            raise ValueError("Unsupported checkpoint format")
        model.eval()
        return model


@torch.no_grad()
def trace_once(model, tok, prompt: str):
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    carry = model.initial_carry(batch)
    # Ask for extra outputs for visualization
    carry, _, _, outs, _ = model(carry=carry, batch=batch, return_keys=["logits", "z_H", "z_L", "steps", "halted"])  # type: ignore
    return outs


def save_visualization(outs, tok, prompt: str, out_path: str):
    logits = outs["logits"].float().cpu()  # [1, T, V]
    z_H = outs.get("z_H")
    steps = outs.get("steps")
    q_steps = steps.item() if torch.is_tensor(steps) else None

    # Token-level confidence: softmax entropy per position
    probs = torch.softmax(logits[0], dim=-1)  # [T, V]
    entropy = (-probs * (probs.clamp_min(1e-9)).log()).sum(-1).numpy()  # [T]

    # Hidden norm
    hid_norm = None
    if z_H is not None:
        zH = z_H.float().cpu()[0]  # [T, H]
        hid_norm = zH.norm(dim=-1).numpy()

    # Plot
    T = logits.shape[1]
    fig, ax = plt.subplots(3 if hid_norm is not None else 2, 1, figsize=(12, 6), sharex=True)
    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]

    ax[0].plot(entropy, label="Token entropy")
    ax[0].set_ylabel("Entropy")
    ax[0].grid(True, alpha=0.3)
    if q_steps is not None:
        ax[0].set_title(f"Prompt len={T}, ACT steps={q_steps}")

    if hid_norm is not None:
        ax[1].plot(hid_norm, color="tab:orange", label="||z_H||")
        ax[1].set_ylabel("Hidden norm")
        ax[1].grid(True, alpha=0.3)
        last_ax = ax[2]
    else:
        last_ax = ax[1]

    # Show top-1 probabilities per position as a proxy for confidence
    top1 = probs.max(dim=-1).values.numpy()
    last_ax.plot(top1, color="tab:green", label="Top-1 prob")
    last_ax.set_ylabel("Top-1 prob")
    last_ax.set_xlabel("Token position")
    last_ax.grid(True, alpha=0.3)

    for a in ax:
        a.legend(loc="upper right")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="Hello world")
    ap.add_argument("--tokenizer_name", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--out", default="outputs/text_trace.png")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = build_model(args.checkpoint, tok)
    outs = trace_once(model, tok, args.prompt)
    save_visualization(outs, tok, args.prompt, args.out)
    print(f"Saved visualization to {args.out}")


if __name__ == "__main__":
    main()

