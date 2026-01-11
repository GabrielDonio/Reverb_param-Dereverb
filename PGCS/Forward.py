import torch
import torchaudio
import matplotlib.pyplot as plt
class ForwardDiffusion:
    def __init__(self, beta_t=None, T=200):
        self.T = T
        if beta_t is None:
            self.beta_t = self.linear_beta_schedule()
        else:
            self.beta_t = beta_t
            
        self.alpha_t = 1.0 - self.beta_t
        
        alpha_cumprod_raw = torch.cumprod(self.alpha_t, dim=0)
        self.alpha_cumprod = torch.cat([torch.tensor([1.0]), alpha_cumprod_raw])

    def forward(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_cumprod = self.alpha_cumprod.to(t.device)
        alpha_bar_t = alpha_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return xt, noise
    def cosine_beta_schedule(self, s=0.008, beta_max=0.005):
        steps = torch.arange(self.T + 1)
        f = torch.cos(((steps / self.T) + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = f / f[0]
        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(beta, 1e-4, beta_max)
    def linear_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02 
        return torch.linspace(beta_start, beta_end, self.T)



if __name__ == "__main__":
    model = ForwardDiffusion()
    x, sr = torchaudio.load("rir_output_50k\\train\\wavs\\rir_000015.wav")
    x = x[0] 
    x = x / torch.max(torch.abs(x))
    N=x.shape[0]
    pure_noise = torch.randn(N)
    plt.plot(pure_noise.cpu().numpy())
    plt.title("Pure Noise")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()
    plt.plot(x.cpu().numpy())
    plt.title("Original Signal x0")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()
    t = 200
    x_t, eps = model.forward(x, torch.tensor([t]))

    if x_t.shape != x.shape:
        print("❌ Shape mismatch between x and x_t")
    else:
        print("✅ Shape OK")

    print(f"x_t shape: {x_t.shape}")
    print(f"epsilon shape: {eps.shape}")
    plt.plot(x_t.cpu().numpy())
    plt.title(f"Noisy Signal x_t at time step t={t}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

