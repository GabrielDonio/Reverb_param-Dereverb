import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class CausalConv2d(nn.Module):
    """Conv2d causale : padding left-only sur T, symétrique sur F."""
    def __init__(self, in_ch, out_ch, kernel_size=(5, 3), dilation=(1, 1), groups=1):
        super().__init__()
        kT, kF = kernel_size
        dT, dF = dilation
        self.pad_T = (kT - 1) * dT
        self.pad_F = ((kF - 1) * dF) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            padding=0, dilation=dilation,
            groups=groups, bias=False
        )

    def forward(self, x):
        x = F.pad(x, (self.pad_F, self.pad_F, self.pad_T, 0))
        return self.conv(x)

class CausalConvNeXtBlock(nn.Module):
    """
    DW-CausalConv (7x5) -> LayerNorm -> PW expand (x4) -> GELU -> PW project
    Residuel inclus.
    """
    def __init__(self, ch, kernel_size=(7, 5), dilation=(1, 1), expansion=4):
        super().__init__()
        self.dw   = CausalConv2d(ch, ch, kernel_size=kernel_size,
                                 dilation=dilation, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1  = nn.Linear(ch, ch * expansion)
        self.pw2  = nn.Linear(ch * expansion, ch)

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)      # (B, T, F, C)
        x = self.norm(x)
        x = self.pw2(F.gelu(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)      # (B, C, T, F)
        return x + residual

class FiLM(nn.Module):
    def __init__(self, time_dim, ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, ch * 2)
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, t_emb):
        gamma, beta = self.proj(t_emb).chunk(2, dim=-1)    # (B, ch) chacun
        return (gamma[:, :, None, None] + 1.0) * x + beta[:, :, None, None]

class GatedSkip(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Conv2d(ch, ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

    def forward(self, skip):
        return torch.sigmoid(self.gate(skip)) * skip

class CausalDown(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.pool   = nn.MaxPool2d(2)
        self.block1 = CausalConvNeXtBlock(in_ch)
        self.proj   = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.block2 = CausalConvNeXtBlock(out_ch)
        self.film   = FiLM(time_dim, out_ch)

    def forward(self, x, t_emb):
        x = self.pool(x)
        x = self.block1(x)
        x = self.proj(x)
        x = self.block2(x)
        return self.film(x, t_emb)


class CausalUp(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.gate_skip = GatedSkip(skip_ch)
        self.proj      = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.block1    = CausalConvNeXtBlock(out_ch)
        self.block2    = CausalConvNeXtBlock(out_ch)
        self.film      = FiLM(time_dim, out_ch)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([self.gate_skip(skip), x], dim=1)
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.film(x, t_emb)

class BottleneckAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads=num_heads,
                                          batch_first=True, dropout=0.0)

    def forward(self, x):
        B, C, T, Fq = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * T, Fq, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        return x_flat.reshape(B, T, Fq, C).permute(0, 3, 1, 2)


class Bottleneck(nn.Module):
    def __init__(self, bot_ch, time_dim, num_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalConvNeXtBlock(bot_ch, dilation=(1,  1)),
            CausalConvNeXtBlock(bot_ch, dilation=(2,  1)),
            CausalConvNeXtBlock(bot_ch, dilation=(4,  1)),
            CausalConvNeXtBlock(bot_ch, dilation=(8,  1)),
            CausalConvNeXtBlock(bot_ch, dilation=(16, 1)),
            CausalConvNeXtBlock(bot_ch, dilation=(32, 1)),
        ])
        self.attn = BottleneckAttention(bot_ch, num_heads=num_heads)
        self.film = FiLM(time_dim, bot_ch)

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x)
        x = self.attn(x)
        return self.film(x, t_emb)


class CausalDereverb(nn.Module):

    def __init__(self, ch=32, time_dim=128, device='cuda'):
        super().__init__()
        self.time_dim = time_dim
        self.device   = device

        bot_ch = ch * 4     # plafonne a ch*4

        # Embedding : sinusoidal -> Linear leger (remplace le t_mlp a 2 couches)
        self.t_proj = nn.Linear(time_dim, time_dim)

        # Entree : concat x_t (canal 0) + reverb (canal 1)
        self.inc = nn.Sequential(
            CausalConv2d(1, ch, kernel_size=(5, 3)),
            nn.GroupNorm(1, ch),
            nn.GELU(),
        )

        # Encodeur — 3 niveaux
        self.down1 = CausalDown(ch,     ch * 2, time_dim)   # T/2, F/2
        self.down2 = CausalDown(ch * 2, ch * 4, time_dim)   # T/4, F/4
        self.down3 = CausalDown(ch * 4, bot_ch, time_dim)   # T/8, F/8

        # Bottleneck
        self.bot = Bottleneck(bot_ch, time_dim, num_heads=4)

        # Decodeur — symetrique
        self.up1 = CausalUp(bot_ch,  ch * 4, ch * 4, time_dim)
        self.up2 = CausalUp(ch * 4,  ch * 2, ch * 2, time_dim)
        self.up3 = CausalUp(ch * 2,  ch,     ch,     time_dim)

        # Dual head
        self.out_mag  = nn.Conv2d(ch, 1, kernel_size=1)
        nn.init.zeros_(self.out_mag.weight);  nn.init.zeros_(self.out_mag.bias)

    def pos_encoding(self, t, channels):
        """Encoding sinusoidal de t dans [0,1] -> (B, channels)."""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x_t, t):
        
        orig_T, orig_F = x_t.shape[2], x_t.shape[3]

        # Pad pour divisibilite par 8
        pad_T = (8 - orig_T % 8) % 8
        pad_F = (8 - orig_F % 8) % 8
        if pad_T > 0 or pad_F > 0:
            x_t    = F.pad(x_t,    (0, pad_F, pad_T, 0))

        # Embedding de t
        t_emb = self.pos_encoding(t.unsqueeze(-1).float(), self.time_dim)  # (B, time_dim)
        t_emb = F.silu(self.t_proj(t_emb))                                 # (B, time_dim)

        # Condition
        x = torch.cat([x_t], dim=1)    # (B, 2, T, F)

        # Encodeur
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)

        # Bottleneck
        x4 = self.bot(x4, t_emb)

        # Decodeur
        x = self.up1(x4, x3, t_emb)
        x = self.up2(x,  x2, t_emb)
        x = self.up3(x,  x1, t_emb)

        # Dual head + blend avec x_t
        out  = self.out_mag(x)
        # Crop
        out = out[:, :, pad_T:pad_T + orig_T, :orig_F]
        return out
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Comparaison des configs :")
    print("-" * 50)
    for ch, td, label in [(24, 128, "leger"), (32, 128, "defaut"), (40, 128, "lourd")]:
        m = CausalDereverb(ch=ch, time_dim=td, device=device).to(device)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  ch={ch:2d}  time_dim={td}  ->  {n:>10,} params  [{label}]")

    print()
    model = CausalDereverb(ch=32, time_dim=128, device=device).to(device)
    B, T, Fq = 4, 375, 256
    x_t    = torch.randn(B, 1, T, Fq).to(device)
    t      = torch.rand(B).to(device)

    out = model(x_t, t)
    assert out.shape == x_t.shape, f"Shape mismatch: {out.shape} vs {x_t.shape}"
    print(f"Input  : {x_t.shape}")
    print(f"Output : {out.shape}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n:,}")
    print("OK")