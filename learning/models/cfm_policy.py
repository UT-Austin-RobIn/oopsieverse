import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.encoder import SegmentationEncoder, StateEncoder
from models.transformer import TransformerLayer
from models.utils import normalize_action, denormalize_action


@dataclass
class PolicyConfig:
    """Configuration for CFM Policy matching B1KDataset setup."""
    # Data dimensions
    action_dim: int = 7  # Franka: [ee_pos, ee_angle, gripper] 
    action_chunk_size: int = 8
    state_dim: int = 23  # Proprio dimension
    
    # Observation setup
    num_seg_views: int = 3  # Number of segmentation camera views
    frame_stack: int = 2  # Number of stacked frames
    seg_img_height: int = 128
    seg_img_width: int = 128
    num_seg_classes: int = 256  # Max segmentation class IDs
    
    # Model architecture
    feature_dim: int = 512 
    num_transformer_blocks: int = 4
    num_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.2
    
    # Segmentation encoder
    seg_embedding_dim: int = 32
    
    # Flow matching
    num_inference_steps: int = 16 
    time_sampler: str = "beta"  # "uniform" or "lognormal" or "beta"
    # Whether to use the one-step reconstruction MSE loss to weigh the flow-matching loss.
    energy_weighted_loss: bool = False

    # Action normalization
    action_min: float = None
    action_max: float = None

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer attention mechanism."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int, batch_size: int, device) -> torch.Tensor:
        """Returns positional encodings of shape (batch_size, seq_len, d_model)"""
        return self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)


def timestep_embedding(t: torch.Tensor, d_model: int, min_period: float = 4e-3, max_period: float = 4.0):
    """Create sinusoidal timestep embeddings for flow matching."""
    if d_model % 2 != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by 2")

    fraction = torch.linspace(0.0, 1.0, d_model // 2, device=t.device)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = t.unsqueeze(-1) / period * 2 * math.pi
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class CFMPolicy(nn.Module):
    """
    Conditional Flow Matching Policy for robotic manipulation.
    
    Inputs:
        - seg_images: Dict[str, Tensor] with shape [B, frame_stack, H, W] (int32 segmentation IDs)
        - state: Tensor [B, frame_stack, state_dim] (proprio)
        - action: Tensor [B, action_chunk, action_dim] (for training)
    
    Output:
        - predicted velocity field for flow matching
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        if config.action_min is not None and config.action_max is not None:
            action_min = torch.tensor(config.action_min)[None, None, :]
            action_max = torch.tensor(config.action_max)[None, None, :]
            self.register_buffer("action_min", action_min)
            self.register_buffer("action_max", action_max)
        self.action_shape = (config.action_chunk_size, config.action_dim)
        self._build_model()

    def _build_model(self, device="cuda"):
        cfg = self.config
        
        # Segmentation encoder (shared across all views)
        self.seg_encoder = SegmentationEncoder(cfg)
        
        # State encoder
        self.state_encoder = StateEncoder(cfg)
        
        # Flow timestep embedding projection
        self.timestep_proj = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.feature_dim),
            nn.SiLU(),
            nn.Linear(cfg.feature_dim, cfg.feature_dim),
        )
        
        # Action projection
        self.action_proj = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.feature_dim),
            nn.SiLU(),
            nn.Linear(cfg.feature_dim, cfg.feature_dim),
        )
        
        # Positional encoding for transformer sequence
        # Sequence: [seg_tokens (num_views * frame_stack), state_token, timestep_token, action_tokens]
        self.num_obs_tokens = cfg.num_seg_views * cfg.frame_stack + 1  # seg views + state
        self.max_seq_len = self.num_obs_tokens + cfg.action_chunk_size # obs + actions
        self.pos_encoding = PositionalEncoding(cfg.feature_dim, self.max_seq_len)
        
        # Transformer decoder
        self.transformer_blocks = nn.ModuleList([
            TransformerLayer(cfg.feature_dim, cfg.ffn_multiplier, cfg.num_heads, dropout_rate=cfg.dropout)
            for _ in range(cfg.num_transformer_blocks)
        ])
        
        # Output projection
        self.action_head = nn.Linear(cfg.feature_dim, cfg.action_dim)

        self.to(device)

    def encode_segmentation(self, seg_images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode segmentation observations.
        
        Args:
            seg_images: Dict mapping view names to tensors of shape [B, frame_stack, H, W]
        
        Returns:
            Tensor of shape [B, num_views * frame_stack, feature_dim]
        """
        tokens = []
        for view_name, seg in seg_images.items():
            B, T, H, W = seg.shape
            # Flatten batch and time: [B*T, H, W]
            seg_flat = seg.reshape(B * T, H, W)
            # Encode: [B*T, feature_dim]
            features = self.seg_encoder(seg_flat)
            # Reshape back: [B, T, feature_dim]
            features = features.reshape(B, T, -1)
            tokens.append(features)
        
        # Concatenate all views: [B, num_views * frame_stack, feature_dim]
        return torch.cat(tokens, dim=1)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode proprioceptive state.
        
        Args:
            state: Tensor of shape [B, state_dim]
        
        Returns:
            Tensor of shape [B, 1, feature_dim] (pooled across frames)
        """
        B, D = state.shape
        # Use only the latest frame for state
        return self.state_encoder(state).unsqueeze(1)  # [B, 1, feature_dim]

    def encode_obs(self, images, state) -> torch.Tensor:
        seg_features = self.encode_segmentation(images)
        state_features = self.encode_state(state)
        # [BS x (num_imgs*img_history_len + 1) x feature_dim]
        return torch.cat([seg_features, state_features], dim=1)
        # return torch.cat(
        #     [seg_features, state_features[:, None, :]], dim=1
        # )  

    def encode_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode flow matching timestep.
        
        Args:
            t: Tensor of shape [B]
        
        Returns:
            Tensor of shape [B, 1, feature_dim]
        """
        t_embed = timestep_embedding(t, self.config.feature_dim)
        return self.timestep_proj(t_embed).unsqueeze(1)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Project noisy action to feature space.
        
        Args:
            action: Tensor of shape [B, action_chunk, action_dim]
        
        Returns:
            Tensor of shape [B, action_chunk, feature_dim]
        """
        return self.action_proj(action)

    def construct_mask(self, device: torch.device) -> torch.Tensor:
        """
        1. Image and state tokens can attend to each other. 
        2. Image and state tokens cannot attend to noisy actions.
        3. Noisy actions can attend to everything else.
        """
        mask = torch.zeros(self.max_seq_len, self.max_seq_len, device=device)
        # Observations + state + timestep cannot attend to noisy actions.
        mask[:self.num_obs_tokens, -self.config.action_chunk_size:] = -torch.inf 
        return mask


    def forward(
        self,
        seg_images: Dict[str, torch.Tensor],
        state: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass predicting velocity field.
        
        Args:
            seg_images: Dict[str, Tensor] with shape [B, frame_stack, H, W]
            state: Tensor [B, frame_stack, state_dim]
            noisy_action: Tensor [B, action_chunk, action_dim]
            t: Tensor [B] flow timestep in [0, 1]
        
        Returns:
            Predicted velocity [B, action_chunk, action_dim]
        """
        B = t.shape[0]
        device = t.device
        
        # Encode observations
        seg_tokens = self.encode_segmentation(seg_images)  # [B, num_views*frame_stack, D]
        state_tokens = self.encode_state(state)  # [B, 1, D]
        t_tokens = self.encode_timestep(t)  # [B, 1, D]
        action_tokens = self.encode_action(noisy_action)  # [B, action_chunk, D]
        # Adding the diffusion step to the action tokens.
        action_tokens += t_tokens
        
        # Build sequence: [seg, state, timestep, action]
        sequence = torch.cat([seg_tokens, state_tokens, action_tokens], dim=1)
        
        # Add positional encoding
        seq_len = sequence.shape[1]
        pos_enc = self.pos_encoding(seq_len, B, device)
        sequence = sequence + pos_enc
        
        # Pass through transformer
        mask = self.construct_mask(device)
        for block in self.transformer_blocks:
            out = block(sequence, attn_mask=mask)
            sequence = out["output"]
        
        # Extract action tokens and predict velocity
        action_output = sequence[:, -self.config.action_chunk_size:]
        velocity = self.action_head(action_output)
        
        return velocity

    def sample_x0(self, action_shape, device, scale: float = 1.0) -> torch.Tensor:
        """Sample initial noise x0."""
        return torch.randn(*action_shape, device=device) * scale

    def compute_loss(
        self,
        seg_images: Dict[str, torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching training loss.
        
        Args:
            seg_images: Dict[str, Tensor] with shape [B, frame_stack, H, W]
            state: Tensor [B, frame_stack, state_dim]
            action: Tensor [B, action_chunk, action_dim] (ground truth)
        
        Returns:
            Scalar loss tensor
        """
        B = action.shape[0]
        device = action.device
        
        # Sample flow timestep
        t = self._sample_timestep(B, device)
        
        # Sample initial noise x0
        x0 = self.sample_x0(action.shape, device)
        
        # Compute interpolated point xt = (1-t)*x0 + t*x1
        t_expanded = t[:, None, None]
        xt = (1 - t_expanded) * x0 + t_expanded * action
        
        # Target velocity is x1 - x0
        target_velocity = action - x0
        
        # Predict velocity
        pred_velocity = self.forward(seg_images, state, xt, t)
        
        # MSE loss
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        return loss

    def _sample_timestep(self, batch_size: int, device) -> torch.Tensor:
        """Sample flow matching timestep."""
        if self.config.time_sampler == "uniform":
            return torch.rand(batch_size, device=device)
        elif self.config.time_sampler == "lognormal":
            t = torch.randn(batch_size, device=device)
            return torch.sigmoid(t)
        elif self.config.time_sampler == "beta":
            t = torch.distributions.beta.Beta(1.5, 1.0).sample(sample_shape=(batch_size,))
            return t.to(device)
        else:
            raise ValueError(f"Unknown time sampler: {self.config.time_sampler}")

    @torch.no_grad()
    def generate_action(
        self,
        seg_images: Dict[str, torch.Tensor],
        state: torch.Tensor,
        n_actions: int = 1,
        init_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate action using Euler integration of the flow ODE.
        
        Args:
            seg_images: Dict[str, Tensor] with shape [B, frame_stack, H, W]
            state: Tensor [B, frame_stack, state_dim]
        
        Returns:
            Generated action [B, action_chunk, action_dim]
        """
        B = state.shape[0]
        device = state.device
        
        # Start from noise
        shape = (B * n_actions, *self.action_shape)
        # x = torch.randn(B * n_actions, *self.action_shape, device=device)
        if init_noise is None:
            x = self.sample_x0(shape, device)
        else:
            x = init_noise.detach().clone()

        # Repeat the state and seg_images for each sample
        state = state.repeat(n_actions, 1)
        seg_images = {k: v.repeat(n_actions, 1, 1, 1) for k, v in seg_images.items()}
        
        # Euler integration
        dt = 1.0 / self.config.num_inference_steps
        for step in range(self.config.num_inference_steps):
            t = torch.full((B * n_actions,), step * dt, device=device)
            velocity = self.forward(seg_images, state, x, t)
            x = x + velocity * dt
        if self.config.action_min is not None and self.config.action_max is not None:
            x = denormalize_action(x, self.action_min, self.action_max)
        if n_actions > 1:
            return x.reshape(B, n_actions, *self.action_shape)
        return x

    def get_action(
        self,
        seg_images: Dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference API matching typical policy interface.
        
        Returns:
            Dict with "action" key containing numpy array
        """
        action = self.generate_action(seg_images, state)
        return {"action": action.cpu().numpy()}

    def get_parameter_count(self) -> Dict:
        """Returns parameter counts for the model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "components": {
                "seg_encoder": sum(p.numel() for p in self.seg_encoder.parameters()),
                "state_encoder": sum(p.numel() for p in self.state_encoder.parameters()),
                "transformer": sum(p.numel() for p in self.transformer_blocks.parameters()),
                "action_head": sum(p.numel() for p in self.action_head.parameters()),
            }
        }

    def print_parameter_summary(self):
        """Print formatted parameter summary."""
        info = self.get_parameter_count()
        print("=" * 50)
        print("CFM Policy Parameter Summary")
        print("=" * 50)
        print(f"Total: {info['total_parameters']/1e6:.2f}M")
        print(f"Trainable: {info['trainable_parameters']/1e6:.2f}M")
        print("\nComponents:")
        for name, count in info["components"].items():
            pct = count / info["total_parameters"] * 100
            print(f"  {name}: {count/1e6:.2f}M ({pct:.1f}%)")
        print("=" * 50)


if __name__ == "__main__":
    from dataset.b1k_dataset import B1KDataset
    from models.trainers.cfm_trainer import CFMTrainer, TrainerConfig
    
    # Create policy
    policy_config = PolicyConfig()
    policy = CFMPolicy(policy_config)
    policy.print_parameter_summary()
    
    # Create dataset
    dataset = B1KDataset(
        data_path="../safe-manipulation-benchmark/resources/playback_data/shelf_place_bad_trajs_playback.hdf5",
        frame_stack=2,
        action_chunk_size=8,
        seg_img_size=(128, 128),
    )
    
    # Test inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    policy.load_state_dict(torch.load("checkpoints/step_5000.pth")["policy_state_dict"])
    policy.eval()
    with torch.no_grad():
        for i in range(1000):
            sample = dataset[i]
            seg_images = {k: v.unsqueeze(0).to(device) for k, v in sample['obs']['extero'].items()}
            state = sample['obs']['proprio'].unsqueeze(0).to(device)
            
            action = policy.generate_action(seg_images, state)
            action_summation = sample['action'][0, :-1].sum()
            # if action_summation > 0:
            print(f"\nGenerated action shape: {action.shape}")
            print("Action summation", action_summation)
            print(f"Action sample:\n{action[0, 0, :]}")
            print(f"GT Action sample:\n{sample['action'][0, :]}")
            print("Loss", torch.norm(action[0] - sample['action'].to("cuda")))
