import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Convolutional residual block with SiLU activation and GroupNorm."""
    
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU()
        
        # Skip connection with projection if channels don't match
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(x + residual)


class SegmentationEncoder(nn.Module):
    """
    Encodes segmentation images (integer class IDs) to feature vectors.
    
    The encoder uses an embedding layer to convert discrete segmentation IDs
    to dense vectors, then processes with a CNN to produce a feature vector.
    
    Input shapes (called from CFMPolicy.encode_segmentation):
        - From dataset: each seg image is [frame_stack, H, W] with dtype int32
        - After batching: [B, frame_stack, H, W]
        - Policy flattens to: [B * frame_stack, H, W] before calling this encoder
    
    Output shape:
        - [B * frame_stack, feature_dim] (one token per frame)
        
    The policy then reshapes to [B, frame_stack, feature_dim] and concatenates
    all views to get [B, num_views * frame_stack, feature_dim] tokens.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.num_seg_classes
        
        # Embedding layer for segmentation class IDs
        # We use num_classes + 1 to handle potential out-of-range IDs (mapped to last index)
        self.embedding = nn.Embedding(
            num_embeddings=config.num_seg_classes + 1,  # +1 for unknown/invalid class
            embedding_dim=config.seg_embedding_dim
        )
        
        # Convolutional encoder
        # Input: [B, seg_embedding_dim, H, W]
        # Using strided convolutions for downsampling
        self.conv_layers = nn.Sequential(
            # Stage 1: embedding_dim -> 64, H/2
            ResidualBlock(config.seg_embedding_dim, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            
            # Stage 2: 64 -> 128, H/4
            ResidualBlock(64, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            
            # Stage 3: 128 -> 256, H/8
            ResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            
            # Stage 4: 256 -> 256, H/16
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            
            # Final residual block
            ResidualBlock(256, 256),
        )
        
        # Average pooling (2, 2, 256) 
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 4))

        self.projection = nn.Sequential(
            nn.Linear(2 * 2 * 256, config.feature_dim),
            nn.SiLU(),
            nn.Linear(config.feature_dim, config.feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Segmentation mask [B, H, W] with integer class IDs (int32 or long)
               Values should be in range [0, num_seg_classes - 1]
        
        Returns:
            Feature vector [B, feature_dim]
        """
        # Clamp segmentation IDs to valid range
        # Map out-of-range values to the last embedding (unknown class)
        x = x.long()
        x = torch.clamp(x, min=0, max=self.num_classes)
        
        # Embed class IDs: [B, H, W] -> [B, H, W, embedding_dim]
        x = self.embedding(x)
        
        # Rearrange to [B, embedding_dim, H, W] for conv layers
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Conv encoding: [B, embedding_dim, H, W] -> [B, 256, H/16, W/16]
        x = self.conv_layers(x)
        
        # Global pooling: [B, 256, H/16, W/16] -> [B, 256, 1, 1] -> [B, 256]
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        
        # Project to feature dim: [B, 256] -> [B, feature_dim]
        x = self.projection(x)
        
        return x


class StateEncoder(nn.Module):
    """
    Encodes proprioceptive state to feature vector.
    
    Input shapes (called from CFMPolicy.encode_state):
        - From dataset: state is [frame_stack, state_dim]
        - After batching: [B, frame_stack, state_dim]
        - Policy extracts latest frame: [B, state_dim] before calling this encoder
    
    Output shape:
        - [B, feature_dim]
        
    The policy then unsqueezes to [B, 1, feature_dim] to use as a token.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.mlp = nn.Sequential(
            nn.Linear(config.state_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.SiLU(),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.SiLU(),
            nn.Linear(config.feature_dim, config.feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State vector [B, state_dim]
        
        Returns:
            Feature vector [B, feature_dim]
        """
        return self.mlp(x)


class PatchSegmentationEncoder(nn.Module):
    """
    Alternative encoder that outputs spatial patch tokens instead of a single pooled vector.
    Useful for finer-grained spatial reasoning. This could be used in the self-attention
    mechanism, but it requires a lot more compute.

    Input: [B, H, W] segmentation mask
    Output: [B, num_patches, feature_dim] where num_patches = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, config, patch_size: int = 16):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.num_classes = config.num_seg_classes

        # Embedding for segmentation IDs
        self.embedding = nn.Embedding(
            num_embeddings=config.num_seg_classes + 1,
            embedding_dim=config.seg_embedding_dim
        )

        # Patch embedding via conv with stride = patch_size
        self.patch_embed = nn.Conv2d(
            config.seg_embedding_dim,
            config.feature_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable position embeddings for patches
        # Will be initialized properly in forward based on actual patch grid size
        self.pos_embed = None

        # Transformer layers for patch processing
        self.norm = nn.LayerNorm(config.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Segmentation mask [B, H, W]
        
        Returns:
            Patch tokens [B, num_patches, feature_dim]
        """
        B, H, W = x.shape

        # Clamp to valid range
        x = torch.clamp(x.long(), min=0, max=self.num_classes)

        # Embed: [B, H, W] -> [B, H, W, embed_dim] -> [B, embed_dim, H, W]
        x = self.embedding(x).permute(0, 3, 1, 2).contiguous()

        # Patchify: [B, embed_dim, H, W] -> [B, feature_dim, H/P, W/P]
        x = self.patch_embed(x)

        # Flatten spatial: [B, feature_dim, H/P, W/P] -> [B, feature_dim, num_patches] -> [B, num_patches, feature_dim]
        x = x.flatten(2).transpose(1, 2)

        # Initialize position embeddings if needed
        num_patches = x.shape[1]
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches, self.config.feature_dim) * 0.02
            ).to(x.device)

        # Add position embeddings
        x = x + self.pos_embed

        # Normalize
        x = self.norm(x)

        return x
