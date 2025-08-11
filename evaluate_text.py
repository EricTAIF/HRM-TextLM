# evaluate_text.py
import argparse, math, torch
from torch.utils.data import DataLoader
from text_dataset import TextBlocks, IGNORE_INDEX

@torch.no_grad()
def eval_ppl(model, loader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].cuda(non_blocking=True) if torch.cuda.is_available() else batch[k]
        
        # Create initial carry
        carry = model.initial_carry(batch)
        
        # Forward pass
        carry, outputs = model(carry=carry, batch=batch)
        logits = outputs["logits"]
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += (batch["labels"] != IGNORE_INDEX).sum().item()
    return math.exp(total_loss / max(1, total_tokens))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    val_set = TextBlocks(args.data_path, "val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    ckpt = torch.load(args.checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model = ckpt["model"] if "model" in ckpt else ckpt
    if torch.cuda.is_available(): model.cuda()

    ppl = eval_ppl(model, val_loader)
    print(f"Perplexity: {ppl:.3f}")

if __name__ == "__main__":
    main()