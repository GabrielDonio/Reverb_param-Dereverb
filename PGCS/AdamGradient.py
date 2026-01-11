import torch
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import matplotlib.pyplot as plt
import soundfile as sf

def conv1d(x, h, mode="same"):
    x = x.unsqueeze(0).unsqueeze(0)
    h = h.unsqueeze(0).unsqueeze(0)
    Lx = x.shape[-1]
    Lh = h.shape[-1]
    pad_left = (Lh - 1) // 2
    pad_right = Lh - 1 - pad_left
    x_padded = F.pad(x, (pad_left, pad_right))
    y = F.conv1d(x_padded, h)
    y = y[..., :Lx]
    return y.squeeze()

# Chargement et préparation des signaux
x, sr = torchaudio.load("clean_trainset_28spk_wav/p226_001.wav")
x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(x)
sr = 16000
h, _ = torchaudio.load("rir_output_50k\\train\\wavs\\rir_000012.wav")
x = x[0]
h = h[0]
h = h / (h.sum() + 1e-8)

y = conv1d(x, h, mode="same")
sigma_noise = 0.0001
y = y + torch.randn_like(y) * sigma_noise

x_est = torch.zeros_like(y, requires_grad=True)

optimizer = torch.optim.Adam([x_est], lr=0.03)
num_iter = 90
lambda_reg = 5e-4

for i in range(num_iter):
    optimizer.zero_grad()

    y_hat = conv1d(x_est, h, mode="same")
    data_loss = torch.mean((y_hat - y) ** 2)

    dx = x_est[1:] - x_est[:-1]
    smooth_loss = lambda_reg * torch.mean(dx ** 2)

    loss = data_loss + smooth_loss
    loss.backward()
    optimizer.step()

    if i % 30 == 0:
        print(f"Iter {i}, Loss: {loss.item():.6f}")
x_rec = x_est.detach()
x_rec = x_rec/(x_rec.abs().max()+1)
x=x/(x.abs().max()+1)
plt.figure(figsize=(12,4))
plt.plot(y.numpy(), label="y (reverb)")
plt.plot(x_rec.numpy(), label="x_rec (Adam)")
plt.plot(x.numpy(), label="x original", alpha=0.5)
plt.legend()
plt.show()

sd.play(y.numpy(), samplerate=sr)
sd.wait()
sd.play(x_rec.numpy(), samplerate=sr)
sd.wait()
sd.play(x.numpy(), samplerate=sr)
sd.wait()
sf.write("reconstructed_signal.wav", x_rec.numpy(), sr)