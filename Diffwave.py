from email.mime import audio
import torch 
import torchaudio
import torch.nn as nn
import math
from DiffusionEmb import DiffusionEmbedding
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(res_channels, 2 * res_channels, kernel_size=3, 
                                      padding=dilation, dilation=dilation)
        
        self.diffusion_projection = nn.Linear(256, res_channels)
        
        self.output_projection = nn.Conv1d(res_channels, res_channels, 1)
        self.skip_projection = nn.Conv1d(res_channels, skip_channels, 1)

    def forward(self, x, diffusion_step):
        # x shape: [B, C, L]
        # diffusion_step shape: [B, 256]
        
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        
        y = self.dilated_conv(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.tanh(gate) * torch.sigmoid(filter)
        
        res = self.output_projection(y)
        skip = self.skip_projection(y)
        
        res = res + x  # residual connection
        return res, skip
class DiffWave(nn.Module):
    def __init__(self, n_layers=25, res_channels=48, skip_channels=32):
        super().__init__()
        self.res_channels = res_channels
        self.skip_channels = skip_channels

        self.input_projection = nn.Conv1d(1, res_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(128)
        
        self.diffusion_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        self.residual_layers = nn.ModuleList([
            ResidualBlock(res_channels, skip_channels, dilation=2**(i % 10))
            for i in range(n_layers)
        ])
        
        self.final_projection = nn.Sequential(
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, 1, 1)
        )

    def forward(self, audio, t):
        # Handle input shapes: accept [B, L] or [B, 1, L] or [B, C, L] -> collapse channels to mono
        was_2d = (audio.dim() == 2)
        if was_2d:
            audio = audio.unsqueeze(1)  # [B, 1, L]
        if audio.dim() == 3 and audio.size(1) != 1:
            audio = audio.mean(dim=1, keepdim=True)

        x = F.relu(self.input_projection(audio))  # [B, res_channels, L]

        # Prepare diffusion embedding
        t_emb = t
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(-1)
        diffusion_step = self.diffusion_mlp(self.diffusion_embedding(t_emb))  # [B, 256]

        # Accumulate skip connections
        skip_acc = torch.zeros((x.size(0), self.skip_channels, x.size(2)), device=x.device, dtype=x.dtype)
        for layer in self.residual_layers:
            x, skip = layer(x, diffusion_step)
            skip_acc = skip_acc + skip

        out = self.final_projection(F.relu(skip_acc))  # [B, 1, L]

        # Always return [B, 1, L] to avoid broadcasting issues with targets shaped [B,1,L]
        # (do not squeeze based on input form)
        return out
        