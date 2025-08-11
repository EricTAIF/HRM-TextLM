"""
Enhanced loss functions V2 with stability improvements and gradient monitoring
"""

from typing import Any, Tuple, Dict, Sequence, Optional
import math

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def stable_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, 
                        ignore_index: int = -100, label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Numerically stable cross-entropy with optional label smoothing
    """
    # Cast to float32 for numerical stability
    logits = logits.to(torch.float32)
    
    if label_smoothing > 0.0:
        # Label smoothing for regularization
        vocab_size = logits.size(-1)
        confidence = 1.0 - label_smoothing
        smoothing_value = label_smoothing / (vocab_size - 1)
        
        # Create one-hot and smooth
        one_hot = torch.zeros_like(logits).scatter(-1, labels.unsqueeze(-1).long(), confidence)
        one_hot += smoothing_value
        
        # Mask ignored indices
        valid_mask = (labels != ignore_index).unsqueeze(-1).float()
        one_hot = one_hot * valid_mask
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(one_hot * log_probs).sum(dim=-1)
        
        # Only keep valid positions
        loss = loss * (labels != ignore_index).float()
        
        return loss
    else:
        # Standard cross-entropy with stability
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]), 
            labels.to(torch.long).view(-1), 
            ignore_index=ignore_index, 
            reduction="none"
        ).view(labels.shape)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, 
               alpha: float = 1.0, gamma: float = 2.0, 
               ignore_index: int = -100) -> torch.Tensor:
    """
    Focal loss for handling class imbalance
    """
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1).long(),
        ignore_index=ignore_index,
        reduction="none"
    )
    
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.view(labels.shape)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute global gradient norm for monitoring"""
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    return math.sqrt(total_norm) if param_count > 0 else 0.0


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients and return the norm before clipping
    """
    # Compute gradient norm before clipping
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    
    # Clip gradients
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    return total_norm


class ACTLossHeadV2(nn.Module):
    """Enhanced ACT loss with stability improvements and gradient monitoring"""
    
    def __init__(self, model: nn.Module, loss_type: str = "stable_cross_entropy", 
                 task: str = "puzzle", **loss_kwargs):
        super().__init__()
        self.model = model
        self.task = getattr(model.config if hasattr(model, 'config') else model.inner.config, 'task', task)
        
        # Loss configuration
        self.loss_kwargs = loss_kwargs
        self.label_smoothing = loss_kwargs.get('label_smoothing', 0.0)
        self.focal_alpha = loss_kwargs.get('focal_alpha', 1.0)
        self.focal_gamma = loss_kwargs.get('focal_gamma', 2.0)
        
        # ACT loss weights (reduced for stability)
        self.act_loss_weight = getattr(model.config if hasattr(model, 'config') else model.inner.config, 
                                      'act_loss_weight', 0.1)
        
        # Gradient monitoring
        self.grad_clip_norm = loss_kwargs.get('grad_clip_norm', 1.0)
        
        # Loss function selection
        if loss_type == "stable_cross_entropy":
            self.loss_fn = stable_cross_entropy
        elif loss_type == "focal_loss":
            self.loss_fn = focal_loss
        else:
            self.loss_fn = globals()[loss_type]
        
        # Moving averages for monitoring
        self.register_buffer("loss_ema", torch.tensor(0.0))
        self.register_buffer("grad_norm_ema", torch.tensor(0.0))
        self.ema_decay = 0.99
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        # Model forward pass
        new_carry, outputs = self.model(**model_kwargs)
        
        if self.task == "text_lm":
            # Enhanced text LM loss computation
            batch = model_kwargs.get('batch', {})
            labels = batch.get("labels")
            if labels is None:
                labels = new_carry.current_data.get("labels")
            
            logits = outputs["logits"]
            
            # Compute main loss with label smoothing
            if self.loss_fn == stable_cross_entropy:
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=IGNORE_LABEL_ID,
                    label_smoothing=self.label_smoothing
                ).mean()
            else:
                loss = self.loss_fn(
                    logits,
                    labels,
                    ignore_index=IGNORE_LABEL_ID,
                    **{k: v for k, v in self.loss_kwargs.items() 
                       if k in ['alpha', 'gamma', 'focal_alpha', 'focal_gamma']}
                ).mean()
            
            # Enhanced metrics for text
            with torch.no_grad():
                valid_mask = (labels != IGNORE_LABEL_ID)
                valid_tokens = valid_mask.sum()
                
                if valid_tokens > 0:
                    # Accuracy
                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == labels) & valid_mask
                    accuracy = correct.sum().float() / valid_tokens.float()
                    
                    # Perplexity
                    token_losses = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=IGNORE_LABEL_ID,
                        reduction="none"
                    )
                    avg_loss = token_losses[token_losses != 0].mean()
                    perplexity = torch.exp(avg_loss)
                    
                    # Gradient monitoring
                    grad_norm = compute_gradient_norm(self.model)
                    
                    # Update EMAs
                    self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * loss.detach()
                    self.grad_norm_ema = self.ema_decay * self.grad_norm_ema + (1 - self.ema_decay) * grad_norm
                    
                    metrics = {
                        "count": torch.tensor(1.0, device=labels.device),
                        "loss": loss.detach(),
                        "accuracy": accuracy,
                        "perplexity": perplexity,
                        "valid_tokens": valid_tokens.float(),
                        "grad_norm": torch.tensor(grad_norm, device=labels.device),
                        "loss_ema": self.loss_ema,
                        "grad_norm_ema": self.grad_norm_ema,
                    }
                else:
                    metrics = {
                        "count": torch.tensor(1.0, device=labels.device),
                        "loss": loss.detach(),
                    }
            
            # Filter outputs for return  
            detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
            
            return new_carry, loss, metrics, detached_outputs, torch.tensor(True)
            
        else:
            # Enhanced puzzle logic with stability improvements
            labels = new_carry.current_data["labels"]

            # Correctness computation
            with torch.no_grad():
                mask = labels != IGNORE_LABEL_ID
                loss_counts = mask.sum(-1)
                loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

                is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
                seq_is_correct = is_correct.sum(-1) == loss_counts
                
                # Enhanced metrics
                valid_metrics = new_carry.halted & (loss_counts > 0)
                
                # Gradient monitoring
                grad_norm = compute_gradient_norm(self.model)
                
                metrics = {
                    "count": valid_metrics.sum(),
                    "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                    "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                    "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                    "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
                    "grad_norm": torch.tensor(grad_norm, device=labels.device),
                }

            # Enhanced loss computation
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
            
            # Stabilized Q-learning losses with reduced weights
            q_halt_loss = F.binary_cross_entropy_with_logits(
                outputs["q_halt_logits"], 
                seq_is_correct.to(outputs["q_halt_logits"].dtype), 
                reduction="sum"
            )

            metrics.update({
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            })

            # Q continue loss with stability improvements
            q_continue_loss = 0
            if "target_q_continue" in outputs:
                # Clip target values for stability
                target_q = outputs["target_q_continue"].clamp(0.01, 0.99)
                q_continue_loss = F.binary_cross_entropy_with_logits(
                    outputs["q_continue_logits"], 
                    target_q, 
                    reduction="sum"
                )
                metrics["q_continue_loss"] = q_continue_loss.detach()

            # Combined loss with reduced ACT weight
            total_loss = lm_loss + self.act_loss_weight * (q_halt_loss + q_continue_loss)

            # Filter outputs for return
            detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

            return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class LossScheduler:
    """Dynamic loss weight scheduling for stability"""
    
    def __init__(self, initial_weight: float = 0.1, min_weight: float = 0.01, 
                 max_weight: float = 1.0, patience: int = 100):
        self.initial_weight = initial_weight
        self.current_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.patience = patience
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def step(self, loss: float) -> float:
        """Update weight based on loss progression"""
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter > self.patience:
            # Reduce weight if loss plateaus
            self.current_weight = max(self.current_weight * 0.9, self.min_weight)
            self.patience_counter = 0
            
        return self.current_weight


class GradientMonitor:
    """Monitor gradient statistics for debugging training instability"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_norms = {}
        self.layer_stats = {}
        
    def update(self):
        """Update gradient statistics"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norm = param.data.norm(2).item()
                
                self.layer_norms[name] = grad_norm
                self.layer_stats[name] = {
                    'grad_norm': grad_norm,
                    'param_norm': param_norm,
                    'ratio': grad_norm / (param_norm + 1e-8)
                }
                
    def get_problematic_layers(self, threshold: float = 10.0) -> Dict[str, float]:
        """Identify layers with large gradient norms"""
        return {name: stats['grad_norm'] 
                for name, stats in self.layer_stats.items() 
                if stats['grad_norm'] > threshold}
                
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        if not self.layer_stats:
            return {}
            
        grad_norms = [stats['grad_norm'] for stats in self.layer_stats.values()]
        ratios = [stats['ratio'] for stats in self.layer_stats.values()]
        
        return {
            'mean_grad_norm': sum(grad_norms) / len(grad_norms),
            'max_grad_norm': max(grad_norms),
            'mean_ratio': sum(ratios) / len(ratios),
            'max_ratio': max(ratios)
        }