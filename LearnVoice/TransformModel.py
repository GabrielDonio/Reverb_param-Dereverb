import torch
import torchaudio
import torchaudio.models.conformer as conformer
import torch.nn as nn

class ConformerModel(nn.Module):
    def __init__(self,freqdim,heads,ffn_dim,depth,conv_kernel_size=41,dropout=0.1):
        super(ConformerModel, self).__init__()
        self.model = conformer.Conformer(
            input_dim=freqdim,
            num_heads=heads,
            ffn_dim=ffn_dim,
            num_layers=depth,
            depthwise_conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        self.proj = nn.Linear(freqdim, freqdim)

    def forward(self, x):
        B,T,_= x.shape
        lengths = torch.full((B,), T, dtype=torch.int64, device=x.device)
        x, _ = self.model(x, lengths)
        mask = self.proj(x)
        mask = torch.sigmoid(mask)
        #mask = torch.relu(mask)
        return mask
if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = ConformerModel(freqdim=256, heads=8, ffn_dim=192, depth=3, conv_kernel_size=41, dropout=0.1)
    print("Number of trainable parameters:", count_parameters(model))
