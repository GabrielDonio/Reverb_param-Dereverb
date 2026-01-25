import torch
import torch.nn as nn

class MediumLightDiscriminator2D(nn.Module):
    def __init__(self):
        super(MediumLightDiscriminator2D, self).__init__()
        
        def block(in_ch, out_ch, stride=2, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, 
                          padding=dilation, dilation=dilation, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            # Entrée : [B, 2, T, F]
            block(2, 48, stride=2),     
            block(48, 192, stride=2),       
            # Couche de sortie
            nn.Conv2d(192, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_gen, x_cond):
        # Concaténation Généré + Original (Condition)
        x = torch.cat([x_gen.unsqueeze(1), x_cond.unsqueeze(1)], dim=1)
        
        out = self.model(x)
        return out.mean(dim=(1, 2, 3))

if __name__ == "__main__":
    model = MediumLightDiscriminator2D()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    
    # Test
    x = torch.randn(2, 137, 256)
    y = torch.randn(2, 137, 256)
    print(f"Output Shape: {model(x, y).shape}") # [2]