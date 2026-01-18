import torch
import torchaudio
from Diffwave import DiffWave
from Forward import ForwardRev
import matplotlib.pyplot as plt


class SamplingDerev:
    def __init__(self, model, forward,delta=0.01):
        self.model = model
        self.forward = forward
        self.delta = delta
        self.esp_t = self.forward.esp_t
    @torch.no_grad()
    def sample(self, y):
        device = y.device
        dtype = y.dtype
        n=torch.randn_like(y)
        xt = y + self.esp_t(torch.tensor(1.0, device=device, dtype=dtype)) * n
        for t in torch.arange(1 - self.delta, 0, -self.delta, device=device, dtype=dtype):
            n = torch.randn_like(y)
            xt = (self.delta/t)*self.model(xt, t) + (1-self.delta/t)*xt 
            n=(t-self.delta)*torch.sqrt(self.esp_t(t-self.delta)**2-self.esp_t(t)**2)*n
            xt = xt + n
        return xt
if __name__ == "__main__":
    model = DiffWave()
    forward = ForwardRev()
    sampler = SamplingDerev(model, forward)
    y,sr = torchaudio.load("reverb_testset_wav\p232_001.wav")
    x_real,sr = torchaudio.load("clean_testset_wav\p232_001.wav") 
    x_real = torchaudio.functional.resample(x_real, sr, 16000)
    y = torchaudio.functional.resample(y, sr, 16000)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    x_rec = sampler.sample(y)
    print(x_rec.shape)
    plt.figure(figsize=(12, 6))
    plt.plot(x_rec.cpu().view(-1).numpy(),alpha=0.3, label="Déréverbéré")
    plt.plot(x_real.cpu().view(-1).numpy(), alpha=0.7, label="Réel")
    plt.title("Déréverbération avec Diffwave")
    plt.legend()
    plt.show()