"""
Enhanced pretraining script V2 with stability improvements
- Improved learning rate scheduling  
- Gradient clipping and monitoring
- Enhanced checkpointing
- Better hyperparameter settings
"""

from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

try:
    from adam_atan2 import AdamATan2
    HAS_ADAM_ATAN2 = True
except ImportError:
    print("Warning: adam_atan2 not available, falling back to standard AdamW")
    from torch.optim import AdamW as AdamATan2
    HAS_ADAM_ATAN2 = False

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from text_dataset import TextBlocks, IGNORE_INDEX
from models.losses_v2 import IGNORE_LABEL_ID, ACTLossHeadV2, GradientMonitor
from utils.functions import load_model_class, get_model_source_path
# Import cosine schedule function from original pretrain
from pretrain import cosine_schedule_with_warmup_lr_lambda


# Enhanced configuration with stability improvements
@dataclass
class PretrainConfigV2:
    # Data
    data_path: str
    task: str = "puzzle"  # "puzzle" or "text_lm"

    # Training hyperparameters - optimized for stability
    global_batch_size: int = 32
    epochs: int = 4
    
    # Enhanced learning rate scheduling
    lr: float = 5e-5  # Reduced from 1e-4 for stability
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 1000
    lr_schedule: str = "cosine_with_restarts"  # Enhanced scheduling
    lr_restart_period: int = 1000
    
    # Enhanced optimizer settings
    beta1: float = 0.9
    beta2: float = 0.999  # Increased from 0.95 for stability
    weight_decay: float = 0.1
    puzzle_emb_weight_decay: float = 0.1
    puzzle_emb_lr: float = 1e-2
    
    # Gradient management
    grad_clip_norm: float = 1.0  # Gradient clipping for stability
    grad_accum_steps: int = 1
    
    # Enhanced loss configuration  
    loss_type: str = "stable_cross_entropy"
    label_smoothing: float = 0.1  # Regularization
    
    # Architecture
    arch: Any = None

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Checkpointing - enhanced
    seed: int = 0
    checkpoint_every_eval: bool = True  # More frequent checkpointing
    checkpoint_every_minutes: Optional[int] = 30  # Every 30 minutes
    eval_interval: Optional[int] = 100  # More frequent evaluation
    eval_save_outputs: List[str] = None
    
    # Monitoring
    wandb_project: Optional[str] = None
    monitor_gradients: bool = True
    log_frequency: int = 10


@dataclass 
class TrainStateV2:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    
    step: int
    total_steps: int
    last_checkpoint_time: float = 0.0
    
    # Enhanced monitoring
    gradient_monitor: Optional[GradientMonitor] = None
    best_loss: float = float('inf')
    patience_counter: int = 0


def create_enhanced_optimizer(config: PretrainConfigV2, model: nn.Module, 
                            world_size: int) -> Sequence[torch.optim.Optimizer]:
    """Create optimizers with enhanced settings"""
    
    # Separate parameters for different learning rates
    puzzle_emb_params = []
    regular_params = []
    
    for name, param in model.named_parameters():
        if 'puzzle_emb' in name:
            puzzle_emb_params.append(param)
        else:
            regular_params.append(param)
    
    optimizers = []
    
    # Main optimizer with enhanced settings
    if regular_params:
        main_optimizer = AdamATan2(
            regular_params,
            lr=config.lr / world_size,  # Scale by world size
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            eps=1e-8,  # Numerical stability
        )
        optimizers.append(main_optimizer)
    
    # Puzzle embedding optimizer
    if puzzle_emb_params:
        puzzle_optimizer = AdamATan2(
            puzzle_emb_params,
            lr=config.puzzle_emb_lr / world_size,
            betas=(config.beta1, config.beta2), 
            weight_decay=config.puzzle_emb_weight_decay,
            eps=1e-8,
        )
        optimizers.append(puzzle_optimizer)
    
    return optimizers


def enhanced_lr_schedule(step: int, config: PretrainConfigV2, total_steps: int) -> float:
    """Enhanced learning rate scheduling with restarts"""
    
    if config.lr_schedule == "cosine_with_restarts":
        # Cosine with warm restarts
        restart_step = step % config.lr_restart_period
        restart_progress = restart_step / config.lr_restart_period
        
        if restart_step < config.lr_warmup_steps:
            # Warmup
            warmup_factor = restart_step / config.lr_warmup_steps
            return config.lr * warmup_factor
        else:
            # Cosine decay
            cosine_progress = (restart_step - config.lr_warmup_steps) / (config.lr_restart_period - config.lr_warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_progress))
            return config.lr * (config.lr_min_ratio + (1 - config.lr_min_ratio) * cosine_factor)
    
    else:
        # Standard cosine schedule
        return cosine_schedule_with_warmup_lr_lambda(
            current_step=step,
            base_lr=config.lr,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=total_steps,
            min_ratio=config.lr_min_ratio
        )


def create_dataloader_v2(config: PretrainConfigV2, split: str, rank: int, world_size: int, **kwargs):
    """Enhanced dataloader creation"""
    if config.task == "text_lm":
        dataset = TextBlocks(config.data_path, split)
        
        # Enhanced sampling for stability
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=(split == "train"),
            drop_last=True  # For consistent batch sizes
        ) if world_size > 1 else None
        
        return DataLoader(
            dataset,
            batch_size=config.global_batch_size // world_size,
            sampler=sampler,
            shuffle=(split == "train" and sampler is None),
            num_workers=2,  # Reduced for stability
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # Better performance
        )
    else:
        # Puzzle dataset logic (simplified)
        raise NotImplementedError("Puzzle dataset not implemented in V2")


def train_batch_v2(config: PretrainConfigV2, train_state: TrainStateV2, batch: Any, 
                  global_batch_size: int, rank: int, world_size: int):
    """Enhanced training step with gradient monitoring"""
    
    # Forward pass
    carry, loss, metrics, _, finished = train_state.model(
        carry=train_state.carry,
        batch=batch,
        return_keys=[]
    )
    
    # Scale loss for gradient accumulation
    scaled_loss = loss / config.grad_accum_steps
    scaled_loss.backward()
    
    # Gradient monitoring
    if config.monitor_gradients and train_state.gradient_monitor is not None:
        train_state.gradient_monitor.update()
    
    # Gradient accumulation
    if (train_state.step + 1) % config.grad_accum_steps == 0:
        # Gradient clipping
        if config.grad_clip_norm > 0:
            total_grad_norm = 0.0
            for optimizer in train_state.optimizers:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
            total_grad_norm = math.sqrt(total_grad_norm)
            
            # Clip gradients
            for optimizer in train_state.optimizers:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], config.grad_clip_norm)
            
            # Add gradient norm to metrics
            if rank == 0:
                metrics = metrics or {}
                metrics["grad_norm"] = torch.tensor(total_grad_norm)
        
        # Optimizer step
        for i, optimizer in enumerate(train_state.optimizers):
            # Update learning rate
            new_lr = enhanced_lr_schedule(train_state.step, config, train_state.total_steps)
            if i == 0:  # Main optimizer
                scaled_lr = new_lr / world_size
            else:  # Puzzle embedding optimizer  
                scaled_lr = config.puzzle_emb_lr / world_size
            
            for group in optimizer.param_groups:
                group['lr'] = scaled_lr
            
            optimizer.step()
            optimizer.zero_grad()
    
    # Update carry
    train_state.carry = carry
    
    # Collect metrics
    if rank == 0 and metrics is not None:
        # Add learning rate to metrics
        metrics["lr"] = torch.tensor(enhanced_lr_schedule(train_state.step, config, train_state.total_steps))
        
        # Add gradient monitoring metrics
        if config.monitor_gradients and train_state.gradient_monitor is not None:
            grad_summary = train_state.gradient_monitor.get_summary()
            for key, value in grad_summary.items():
                metrics[f"grad_{key}"] = torch.tensor(value)
    
    return metrics


@hydra.main(config_path="config", config_name="cfg_pretrain_v2", version_base=None)
def launch_v2(hydra_config: DictConfig):
    """Enhanced training launcher"""
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

    # Setup device
    if torch.cuda.is_available():
        torch.cuda.set_device(RANK % torch.cuda.device_count())
        device = torch.device(f"cuda:{RANK}")
    else:
        device = torch.device("cpu")

    print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Using device {device}")

    # Create config
    config_dict = dict(hydra_config)
    if config_dict.get('eval_save_outputs') is None:
        config_dict['eval_save_outputs'] = []
    config = PretrainConfigV2(**config_dict)
    
    # Enhanced project naming
    if config.project_name is None:
        config.project_name = f"{os.path.basename(config.data_path).capitalize()} HRM-V2"
    if config.run_name is None:
        config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
    if config.checkpoint_path is None:
        config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

    # Initialize W&B
    if RANK == 0:
        wandb.init(
            project=config.wandb_project or config.project_name,
            name=config.run_name,
            config=config.__dict__,
            tags=["hrm-v2", "stability", config.task]
        )

    # Load model
    model_cls = load_model_class(config.arch.name)
    
    # Enhanced model configuration
    model_config = {
        **config.arch,
        "task": config.task,
        "gradient_checkpointing": True,
        "act_loss_weight": 0.1,
        "use_peri_ln": True,
    }
    
    if config.task == "text_lm":
        # Load text metadata
        import json
        with open(os.path.join(config.data_path, "meta.json"), "r") as f:
            meta = json.load(f)
        
        model_config.update({
            "batch_size": config.global_batch_size,
            "seq_len": meta["block_size"],
            "vocab_size": meta["vocab_size"],
            "pad_token_id": meta["pad_token_id"],
            "num_puzzle_identifiers": 1,  # Dummy
            "puzzle_emb_ndim": 0,
        })

    # Create model
    model = model_cls(model_config)
    model = model.to(device)
    
    # Enhanced loss with monitoring
    loss_model = ACTLossHeadV2(
        model, 
        config.loss_type,
        config.task,
        label_smoothing=config.label_smoothing,
        grad_clip_norm=config.grad_clip_norm
    )
    loss_model = loss_model.to(device)

    # Model compilation
    if hasattr(torch, 'compile'):
        try:
            loss_model = torch.compile(loss_model)
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")

    # Create optimizers
    optimizers = create_enhanced_optimizer(config, loss_model, WORLD_SIZE)

    # Create dataloaders
    train_loader = create_dataloader_v2(config, "train", RANK, WORLD_SIZE)
    eval_loader = create_dataloader_v2(config, "val", RANK, WORLD_SIZE)

    # Calculate total steps
    total_steps = config.epochs * len(train_loader)

    # Initialize training state
    train_state = TrainStateV2(
        model=loss_model,
        optimizers=optimizers,
        optimizer_lrs=[config.lr, config.puzzle_emb_lr] if len(optimizers) > 1 else [config.lr],
        carry=loss_model.initial_carry(next(iter(train_loader))),
        step=0,
        total_steps=total_steps,
        last_checkpoint_time=time.time(),
        gradient_monitor=GradientMonitor(loss_model) if config.monitor_gradients else None
    )

    # Training loop with enhanced monitoring
    for epoch in range(config.epochs):
        if RANK == 0:
            print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {epoch}")

        train_state.model.train()
        
        # Progress bar
        pbar = tqdm.tqdm(train_loader, disable=(RANK != 0))
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            # Training step
            metrics = train_batch_v2(config, train_state, batch, config.global_batch_size, RANK, WORLD_SIZE)
            
            # Update step counter
            train_state.step += 1
            
            # Logging
            if RANK == 0 and metrics is not None and train_state.step % config.log_frequency == 0:
                # Convert metrics to scalars
                log_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        log_metrics[f"train/{k}"] = v.item()
                    else:
                        log_metrics[f"train/{k}"] = v
                
                wandb.log(log_metrics, step=train_state.step)
                
                # Update progress bar
                if "loss" in log_metrics:
                    pbar.set_description(f"Loss: {log_metrics['train/loss']:.4f}")
            
            # Evaluation and checkpointing
            if config.eval_interval and train_state.step % config.eval_interval == 0:
                # Evaluation
                train_state.model.eval()
                eval_metrics = {}  # Implement evaluation logic
                
                if RANK == 0:
                    wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=train_state.step)
                
                train_state.model.train()
            
            # Enhanced checkpointing
            current_time = time.time()
            time_based_checkpoint = (
                config.checkpoint_every_minutes is not None and
                (current_time - train_state.last_checkpoint_time) >= (config.checkpoint_every_minutes * 60)
            )
            
            if RANK == 0 and (config.checkpoint_every_eval or time_based_checkpoint):
                # Save checkpoint
                os.makedirs(config.checkpoint_path, exist_ok=True)
                checkpoint_file = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt")
                
                torch.save({
                    'model_state_dict': train_state.model.state_dict(),
                    'optimizer_state_dicts': [opt.state_dict() for opt in train_state.optimizers],
                    'step': train_state.step,
                    'config': config,
                    'metrics': metrics
                }, checkpoint_file)
                
                train_state.last_checkpoint_time = current_time
                
                if time_based_checkpoint:
                    print(f"Time-based checkpoint saved at step {train_state.step}")

    # Final checkpoint
    if RANK == 0:
        final_checkpoint = os.path.join(config.checkpoint_path, "final_model.pt")
        torch.save(train_state.model.state_dict(), final_checkpoint)
        print(f"Final model saved to {final_checkpoint}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch_v2()