import torch.nn as nn
import torch
import numpy as np


class Qwen2_5_VisionPatchEmbed_conv3d(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
    
    
class Qwen2_5_VisionPatchEmbed_conv2d(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Conv2d replaces Conv3d: input channels = in_channels * temporal_patch_size
        self.proj = nn.Conv2d(
            in_channels * temporal_patch_size,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def load_conv3d_weight(self, conv3d_weight: torch.Tensor):
        """
        Convert Conv3d weights [E, C, T, H, W] to Conv2d weights [E, C*T, H, W]
        """
        E, C, T, H, W = conv3d_weight.shape
        assert C == self.in_channels
        assert T == self.temporal_patch_size
        # Match the channel order used when flattening the input tensor
        conv2d_weight = conv3d_weight.permute(0, 2, 1, 3, 4).reshape(E, C * T, H, W)
        with torch.no_grad():
            self.proj.weight.copy_(conv2d_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )

        B, C, T, H, W = hidden_states.shape
        assert T == self.temporal_patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        hidden_states = hidden_states.reshape(B, C * T, H, W)  # [B, C*T, H, W]

        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))  # [B, E, H_out, W_out]
        hidden_states = hidden_states.flatten(2).transpose(1, 2)         # [B, H_out*W_out, E]
        hidden_states = hidden_states.reshape(-1, self.embed_dim)       # [B*patches, E]

        return hidden_states
    
    
# Test
patch_size = 14
temporal_patch_size = 2
in_channels = 3
embed_dim = 1152

model_conv3d = Qwen2_5_VisionPatchEmbed_conv3d(patch_size, temporal_patch_size, in_channels, embed_dim)
model_conv2d = Qwen2_5_VisionPatchEmbed_conv2d(patch_size, temporal_patch_size, in_channels, embed_dim)

# Simulate Conv3d weights and use the same weights for both models
conv3d_weight = torch.randn(embed_dim, in_channels, temporal_patch_size, patch_size, patch_size)
with torch.no_grad():
    model_conv3d.proj.weight.copy_(conv3d_weight)
model_conv2d.load_conv3d_weight(conv3d_weight)

# Input: [B*T, C, H, W] = [1*2, 3, 14, 14]
input_tensor = torch.randn(2, 3, 14, 14)
output_conv3d = model_conv3d(input_tensor)
output_conv2d = model_conv2d(input_tensor)

diff = np.abs(output_conv2d.data.cpu().numpy() - output_conv3d.data.cpu().numpy())
print("max diff:", diff.max())