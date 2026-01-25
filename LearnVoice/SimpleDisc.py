import torch
import torch.nn as nn
import torchaudio.models.conformer as conformer

class Pix2pixDiscriminator(nn.Module):
    def __init__(self, freqdim=256, heads=4, ffn_dim=32, depth=1, conv_kernel_size=5, dropout=0.1):
        super(Pix2pixDiscriminator, self).__init__()
        self.input_proj = nn.Linear(2*freqdim, freqdim)
        self.model = conformer.Conformer(
            input_dim=freqdim,
            num_heads=heads,
            ffn_dim=ffn_dim,
            num_layers=depth,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        # Sortie : une valeur par frame temporelle (PatchGAN 1D)
        self.proj = nn.Linear(freqdim, 1)  

    def forward(self, x_gen, x_cond):
        x = torch.cat([x_gen, x_cond], dim=-1)  
        x = self.input_proj(x)
        
        B, T, F = x.shape
        lengths = torch.full((B,), T, dtype=torch.int64, device=x.device)
        
        x, _ = self.model(x, lengths) 
        x = self.proj(x) # [B, T, 1]
        
        # On ne pool pas ! On rend la moyenne des scores de chaque frame
        # Cela force G à être réaliste sur chaque instant T.
        return x.mean(dim=1) # [B, 1]
if __name__ == "__main__":
    model = Pix2pixDiscriminator()
    x = torch.randn(4, 139, 256)
    y = torch.randn(4, 139, 256)
    out = model(x, y)
    print(out.shape) 
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print("Number of trainable parameters:", count_parameters(model))
