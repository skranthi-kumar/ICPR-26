import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings with dynamic sizing support"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # Calculate initial number of patches
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Handle variable input sizes
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size ({H}, {W}) not divisible by patch size {self.patch_size}"
        
        # Dynamic patch calculation
        self.current_n_patches = (H // self.patch_size) * (W // self.patch_size)
        
        x = self.proj(x)  # Shape: (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with optional cross-attention support"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Shape: (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # Shape: (B, N, embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x

class MLP(nn.Module):
    """Enhanced feed-forward network with configurable activation"""
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.0, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) regularization"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class TransformerBlock(nn.Module):
    """Enhanced transformer encoder block with attention output option"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0, 
                 drop_path=0.0, qkv_bias=True, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout, qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout, act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_weights
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

def interpolate_pos_embed(pos_embed_old: torch.Tensor, num_patches_new: int, 
                         patch_size: int = 16) -> torch.Tensor:
    """
    Interpolate positional embeddings for different input sizes.
    Handles the transition from DINO's 224x224 to arbitrary sizes.
    """
    # pos_embed_old shape: (1, N_old + 1, D)
    cls_pos_embed = pos_embed_old[:, 0:1, :]  # CLS token
    patch_pos_embed = pos_embed_old[:, 1:, :]  # Patch embeddings
    
    num_patches_old = patch_pos_embed.shape[1]
    
    if num_patches_old == num_patches_new:
        return pos_embed_old
    
    # Calculate grid sizes
    grid_size_old = int(math.sqrt(num_patches_old))
    grid_size_new = int(math.sqrt(num_patches_new))
    
    print(f"Interpolating position embeddings: {grid_size_old}x{grid_size_old} -> {grid_size_new}x{grid_size_new}")
    
    # Reshape to 2D grid for interpolation
    embed_dim = patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.reshape(1, grid_size_old, grid_size_old, embed_dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)  # (1, D, H, W)
    
    # Interpolate
    patch_pos_embed = F.interpolate(
        patch_pos_embed, 
        size=(grid_size_new, grid_size_new), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Reshape back
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)  # (1, H, W, D)
    patch_pos_embed = patch_pos_embed.reshape(1, num_patches_new, embed_dim)
    
    # Concatenate CLS and patch embeddings
    return torch.cat([cls_pos_embed, patch_pos_embed], dim=1)

class VisionTransformerBackbone(nn.Module):
    """
    Enhanced Vision Transformer backbone for counting applications
    Stage 1 of the proposed architecture with support for:
    - DINO pre-training with automatic position embedding interpolation
    - Multi-scale input handling with efficient caching
    - Feature extraction at multiple levels
    - Cross-attention compatibility for downstream stages
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_heads = num_heads
        self.depth = depth
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        
        # Position embedding cache for efficient multi-scale processing
        self.pos_embed_cache = {}
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Learnable tokens and embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                qkv_bias=qkv_bias,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights following ViT conventions"""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def _get_pos_embed_for_size(self, H: int, W: int) -> torch.Tensor:
        """Get position embeddings for given input size with caching"""
        if H == self.img_size and W == self.img_size:
            return self.pos_embed
        
        cache_key = (H, W)
        if cache_key not in self.pos_embed_cache:
            new_num_patches = (H // self.patch_size) * (W // self.patch_size)
            self.pos_embed_cache[cache_key] = interpolate_pos_embed(
                self.pos_embed, new_num_patches, self.patch_size
            ).to(self.pos_embed.device)
        return self.pos_embed_cache[cache_key]
    
    def clear_pos_embed_cache(self):
        """Clear position embedding cache to free memory"""
        self.pos_embed_cache.clear()
    
    def load_dino_pretrained(self, pretrained_path: str, strict: bool = False):
        """
        Load DINO pretrained weights with automatic position embedding interpolation
        """
        try:
            print(f"Loading DINO weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'student' in checkpoint:
                state_dict = checkpoint['student']
            elif 'teacher' in checkpoint:
                state_dict = checkpoint['teacher']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Clean up keys (remove module prefix if present)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # Remove 'module.' prefix
                if k.startswith('backbone.'):
                    k = k[9:]  # Remove 'backbone.' prefix
                cleaned_state_dict[k] = v
            
            # Handle position embedding interpolation
            if 'pos_embed' in cleaned_state_dict:
                pretrained_pos_embed = cleaned_state_dict['pos_embed']
                current_num_patches = self.patch_embed.n_patches
                pretrained_num_patches = pretrained_pos_embed.shape[1] - 1
                
                if pretrained_num_patches != current_num_patches:
                    print(f"Interpolating position embeddings: {pretrained_num_patches} -> {current_num_patches} patches")
                    cleaned_state_dict['pos_embed'] = interpolate_pos_embed(
                        pretrained_pos_embed, current_num_patches, self.patch_size
                    )
            
            # Load weights
            missing_keys, unexpected_keys = self.load_state_dict(cleaned_state_dict, strict=strict)
            
            # Clear cache after loading new weights
            self.clear_pos_embed_cache()
            
            print("DINO weights loaded successfully!")
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                
        except Exception as e:
            print(f"Error loading DINO pretrained weights: {e}")
            raise e
    
    def forward(self, x: torch.Tensor, return_all_tokens: bool = False, 
               return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass with flexible output options
        
        Args:
            x: Input tensor (B, C, H, W)
            return_all_tokens: If True, return both CLS and patch tokens
            return_attention: If True, return attention weights from last layer
        
        Returns:
            Features tensor or tuple with attention weights
        """
        B, C, H, W = x.shape
        
        # Get cached position embeddings
        current_pos_embed = self._get_pos_embed_for_size(H, W)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embeddings
        x = x + current_pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks with optional gradient checkpointing
        attention_weights = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                # Return attention from last layer
                x, attention_weights = block(x, return_attention=True)
            else:
                if self.enable_gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        if return_attention:
            if return_all_tokens:
                return x, attention_weights
            else:
                return x[:, 1:], attention_weights  # Remove CLS token
        
        if return_all_tokens:
            return x
        else:
            return x[:, 1:]  # Remove CLS token, return only patch features
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature maps (patch tokens only)"""
        return self.forward(x, return_all_tokens=False)
    
    def get_cls_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features"""
        features = self.forward(x, return_all_tokens=True)
        return features[:, 0]  # Return CLS token
    
    def get_intermediate_features(self, x: torch.Tensor, layer_indices: list = None) -> Dict[int, torch.Tensor]:
        """
        Extract features from intermediate layers for multi-scale processing
        Useful for prototype generation and hierarchical processing
        """
        if layer_indices is None:
            layer_indices = [3, 6, 9, 11]  # Default intermediate layers
        
        B, C, H, W = x.shape
        
        # Get cached position embeddings (calculated once)
        current_pos_embed = self._get_pos_embed_for_size(H, W)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token and position embeddings once
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + current_pos_embed
        x = self.pos_dropout(x)
        
        # Forward through blocks and collect intermediate features
        features = {}
        for i, block in enumerate(self.blocks):
            if self.enable_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            
            if i in layer_indices:
                features[i] = self.norm(x)[:, 1:]  # Normalize and remove CLS token
        
        return features
    
    def get_multiscale_features(self, x: torch.Tensor, scales: list = [0.5, 1.0, 1.5]) -> Dict[float, torch.Tensor]:
        """
        Extract features at multiple scales for robust counting
        Useful for handling objects of different sizes
        """
        features = {}
        original_size = x.shape[-1]
        
        for scale in scales:
            if scale == 1.0:
                # Use original image
                scaled_x = x
            else:
                # Interpolate to new size
                new_size = int(original_size * scale)
                # Ensure size is divisible by patch_size
                new_size = (new_size // self.patch_size) * self.patch_size
                if new_size < self.patch_size:
                    new_size = self.patch_size
                
                scaled_x = F.interpolate(x, size=(new_size, new_size), 
                                       mode='bilinear', align_corners=False)
            
            features[scale] = self.forward(scaled_x)
        
        return features
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention maps from a specific layer for visualization
        """
        B, C, H, W = x.shape
        current_pos_embed = self._get_pos_embed_for_size(H, W)
        
        # Patch embedding and add position embeddings
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + current_pos_embed
        x = self.pos_dropout(x)
        
        # Forward through blocks
        target_layer = len(self.blocks) + layer_idx if layer_idx < 0 else layer_idx
        
        for i, block in enumerate(self.blocks):
            if i == target_layer:
                x, attention = block(x, return_attention=True)
                return attention
            else:
                x = block(x)
        
        raise ValueError(f"Layer index {layer_idx} out of range")

# Factory functions for different model sizes
def create_vit_small(pretrained_path: Optional[str] = None, img_size: int = 224, 
                    enable_gradient_checkpointing: bool = False) -> VisionTransformerBackbone:
    """Create ViT-Small model"""
    model = VisionTransformerBackbone(
        img_size=img_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    if pretrained_path:
        model.load_dino_pretrained(pretrained_path)
    return model

def create_vit_base(pretrained_path: Optional[str] = None, img_size: int = 224,
                   enable_gradient_checkpointing: bool = False) -> VisionTransformerBackbone:
    """Create ViT-Base model"""
    model = VisionTransformerBackbone(
        img_size=img_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    if pretrained_path:
        model.load_dino_pretrained(pretrained_path)
    return model

def create_vit_large(pretrained_path: Optional[str] = None, img_size: int = 224,
                    enable_gradient_checkpointing: bool = False) -> VisionTransformerBackbone:
    """Create ViT-Large model"""
    model = VisionTransformerBackbone(
        img_size=img_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    if pretrained_path:
        model.load_dino_pretrained(pretrained_path)
    return model

# Testing and demonstration
if __name__ == "__main__":
    print("Testing Optimized ViT Backbone for Counting Architecture...")
    print("=" * 60)
    
    # Test different model sizes
    models = {
        'ViT-Small': create_vit_small(),
        'ViT-Base': create_vit_base(),
    }
    
    # Test inputs of different sizes
    test_inputs = [
        ("224x224", torch.randn(2, 3, 224, 224)),
        ("384x384", torch.randn(2, 3, 384, 384)),
        ("512x512", torch.randn(2, 3, 512, 512)),
    ]
    
    for model_name, model in models.items():
        print(f"\n{model_name} Testing:")
        print("-" * 30)
        
        for size_name, x in test_inputs:
            try:
                # Test basic forward pass
                features = model(x)
                cls_feats = model.get_cls_features(x)
                spatial_feats = model.get_feature_maps(x)
                
                print(f"  {size_name}:")
                print(f"    Full features: {features.shape}")
                print(f"    CLS features: {cls_feats.shape}")
                print(f"    Spatial features: {spatial_feats.shape}")
                
                # Test intermediate features (should show single interpolation)
                intermediate = model.get_intermediate_features(x)
                print(f"    Intermediate layers: {list(intermediate.keys())}")
                
                # Test cache efficiency
                print("    Testing cache efficiency...")
                _ = model.get_intermediate_features(x)  # Should use cached embeddings
                
            except Exception as e:
                print(f"  {size_name}: Error - {e}")
    
    # Test advanced features
    print(f"\nAdvanced Features Test:")
    print("-" * 30)
    model = create_vit_base()
    x = torch.randn(1, 3, 224, 224)
    
    # Test attention visualization
    features, attention = model(x, return_attention=True)
    print(f"Attention visualization - Features: {features.shape}, Attention: {attention.shape}")
    
    # Test multiscale features
    multiscale_feats = model.get_multiscale_features(x, scales=[0.5, 1.0])
    print(f"Multiscale features:")
    for scale, feat in multiscale_feats.items():
        print(f"  Scale {scale}: {feat.shape}")
    
    # Test attention maps
    attn_maps = model.get_attention_maps(x, layer_idx=-1)
    print(f"Attention maps from last layer: {attn_maps.shape}")
    
    # Test cache clearing
    print(f"Cache size before clearing: {len(model.pos_embed_cache)}")
    model.clear_pos_embed_cache()
    print(f"Cache size after clearing: {len(model.pos_embed_cache)}")
    
    print("\nOptimized backbone testing complete!")
    print("✅ Position embedding caching implemented")
    print("✅ Gradient checkpointing support added")
    print("✅ Advanced feature extraction methods available")
    print("Ready for integration with prototype generation stage!")