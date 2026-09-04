from __future__ import annotations

import os
import subprocess
# from dataset.b1k_dataset import B1KDataset
# from models.utils import denormalize_action
from dataclasses import asdict, dataclass
from typing import Annotated, Optional

import torch
import tyro
from torch.utils.data import DataLoader
from tqdm import tqdm
from learning.utils import denormalize_action
from learning.models import CFMPolicy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainerConfig:
    """Configuration for training a CFM policy."""

    # Data
    data_path: Annotated[str, tyro.conf.Positional]
    frame_stack: int = 2
    action_chunk: int = 8
    seg_img_size: int = 128
    normalize_action: bool = True

    # Training
    num_steps: int = 1000
    batch_size: int = 64 
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    
    # Logging
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 500
    
    # Paths
    save_dir: str = "checkpoints"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "oopsieverse"
    wandb_entity: Optional[str] = None 
    wandb_run_name: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint resume
    resume: Optional[str] = None


class CFMTrainer:
    """Minimal trainer for CFM policy with wandb logging."""
    
    def __init__(
        self,
        config: TrainerConfig,
        policy: CFMPolicy,
        train_dataset,
        eval_dataset=None,
        args_dict=None,
    ):
        self.config = config
        self.policy = policy.to(config.device)
        self.train_dataset = train_dataset
        if eval_dataset is None:
            self.eval_dataset = self.train_dataset 
        else:
            self.eval_dataset = eval_dataset
        
        # Setup dataloader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Setup directories
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Setup wandb
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            # Use all args if provided, otherwise use minimal config
            wandb_config = args_dict if args_dict is not None else {
                "num_steps": config.num_steps,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            }
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=wandb_config,
            )
    
    def train(self, resume_step: int = 0):
        """Main training loop."""
        cfg = self.config
        self.policy.train()
        
        step = resume_step - 1 
        running_loss = 0.0
        
        pbar = tqdm(initial=resume_step, total=cfg.num_steps, desc="Training")
        
        while step < cfg.num_steps:
            for batch in self.train_loader:
                if step >= cfg.num_steps:
                    break
                
                # Move to device
                seg_images = {k: v.to(cfg.device) for k, v in batch['obs']['extero'].items()}
                state = batch['obs']['proprio'].to(cfg.device)
                action = batch['action'].to(cfg.device)
                
                # Forward
                self.optimizer.zero_grad()
                loss = self.policy.compute_loss(seg_images, state, action)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
                self.optimizer.step()
                
                # Logging
                running_loss += loss.item()
                step += 1
                pbar.update(1)
                
                if step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.log_every
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    
                    if self.wandb_run:
                        wandb.log({"train/loss": avg_loss, "step": step})
                    
                    running_loss = 0.0
                
                # Save checkpoint
                if step % cfg.save_every == 0:
                    self.save_checkpoint(f"{cfg.save_dir}/step_{step}.pth")
                
                # Evaluation
                if step % cfg.eval_every == 0 and self.eval_dataset is not None:
                    eval_loss = self.evaluate()
                    if self.wandb_run:
                        wandb.log({"eval/mse_loss": eval_loss, "step": step})
        
        pbar.close()
        
        # Save final checkpoint
        self.save_checkpoint(f"{cfg.save_dir}/final.pth")
        
        if self.wandb_run:
            wandb.finish()
        
        print("Training complete!")
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on eval dataset."""
        if self.eval_dataset is None:
            return 0.0
        
        self.policy.eval()
        
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in eval_loader:
            seg_images = {k: v.to(self.config.device) for k, v in batch['obs']['extero'].items()}
            state = batch['obs']['proprio'].to(self.config.device)
            action = batch['action'].to(self.config.device)
            if self.eval_dataset.normalize_action:
                action = denormalize_action(action, self.policy.action_min, self.policy.action_max)
            
            mse_loss = torch.mean((action - self.policy.generate_action(seg_images, state)) ** 2)
            total_loss += mse_loss
            num_batches += 1
            
            if num_batches >= 10:  # Limit eval batches for speed
                break
        
        self.policy.train()
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint: {path}")

def parse_args() -> TrainerConfig:
    """Parse command-line arguments with Tyro."""
    return tyro.cli(TrainerConfig)


if __name__ == "__main__":
    
    args = parse_args()

    from learning.dataset import B1KDataset
    from learning.models import CFMPolicy, PolicyConfig
    
    # Convert args to dictionary for wandb logging
    args_dict = asdict(args)

    # Comment this out if you want to use all objects
    objects_of_interest = ["box_of_crackers", "book", "bottle_of_wine", "bottle_of_beer", "stand"]

    # Create dataset
    dataset = B1KDataset(
        data_path=args.data_path,
        frame_stack=args.frame_stack,
        action_chunk_size=args.action_chunk,
        seg_img_size=(args.seg_img_size, args.seg_img_size),
        normalize_action=args.normalize_action,
        objects_of_interest=objects_of_interest,
    )

    # Create policy
    if args.normalize_action:
        policy_config = PolicyConfig(action_min=dataset.action_min, 
                                     action_max=dataset.action_max)
        policy = CFMPolicy(policy_config)
    else:
        policy = CFMPolicy(PolicyConfig())
    policy.print_parameter_summary()

    print(f"Dataset size: {len(dataset)}")
    # Create trainer
    trainer = CFMTrainer(
        config=args,
        policy=policy,
        train_dataset=dataset,
        args_dict=args_dict,
    )
    
    # Resume from checkpoint if specified
    resume_step = 0
    if args.resume:
        trainer.load_checkpoint(args.resume)
        resume_step = int(args.resume.split("/")[-1].split("_")[-1].split(".")[0])
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Train
    trainer.train(resume_step=resume_step)
