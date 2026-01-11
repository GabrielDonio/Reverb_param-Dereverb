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
        
        # Output projections
        self.output_projection = nn.Conv1d(res_channels, res_channels, 1)
        self.skip_projection = nn.Conv1d(res_channels, skip_channels, 1)

    def forward(self, x, diffusion_step):
        # x shape: [B, C, L]
        # diffusion_step shape: [B, 512]
        
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        
        y = self.dilated_conv(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.tanh(gate) * torch.sigmoid(filter)
        
        res = self.output_projection(y)
        skip = self.skip_projection(y)
        
        return (x + res) * math.sqrt(0.5), skip
class DiffWave(nn.Module):
    def __init__(self, n_layers=20, res_channels=32, skip_channels=32):
        super().__init__()
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
        if audio.dim() == 2:
            x = audio.unsqueeze(1)
        elif audio.dim() == 3:
            x = audio
        else:
            raise ValueError("audio must be 2D or 3D tensor")
        x = F.relu(self.input_projection(x))
        
        diffusion_step = self.diffusion_mlp(self.diffusion_embedding(t))
        
        skip_connection = 0
        for layer in self.residual_layers:
            x, skip = layer(x, diffusion_step)
            skip_connection = skip_connection + skip
            
        x = skip_connection * math.sqrt(1.0 / len(self.residual_layers))
        x = self.final_projection(x)
        return x.squeeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = DiffWave()
print(f"Le modèle a {count_parameters(model):,} paramètres entraînables.")