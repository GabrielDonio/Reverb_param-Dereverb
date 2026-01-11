import torch
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import matplotlib.pyplot as plt

def conv1d_full(x, h):
    x = x.unsqueeze(0).unsqueeze(0)
    h = h.unsqueeze(0).unsqueeze(0)
    padding = h.shape[-1] - 1
    y = F.conv1d(x, h, padding=padding)
    return y.squeeze()

def reverse_kernel(h):
    return torch.flip(h, dims=[0])


def A_operator(x, h, lambda_reg):
    y = conv1d_full(x, h)
    ht = reverse_kernel(h)
    y = conv1d_full(y, ht)
    return y + lambda_reg * x

def conjugate_gradient(y, h, num_iter=80, lambda_reg=1e-3, eps=1e-8):
    Lx = y.shape[0] - h.shape[0] + 1
    device = y.device

    def H(x): return conv1d_full(x, h)
    def HT(z): return conv1d_full(z, reverse_kernel(h))[:Lx]

    x = torch.zeros(Lx, device=device)
    b = HT(y)
    r = b - (HT(H(x)) + lambda_reg * x)
    p = r.clone()

    for k in range(num_iter):
        Ap = HT(H(p)) + lambda_reg * p
        alpha = torch.dot(r, r) / (torch.dot(p, Ap) + eps)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = torch.dot(r_new, r_new) / (torch.dot(r, r) + eps)
        p = r_new + beta * p
        r = r_new

        if k % 20 == 0:
            loss = torch.mean((H(x) - y)**2) + lambda_reg * torch.mean(x**2)
            print(f"Iter {k}, Loss {loss.item():.6f}")

    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load clean signal
x, sr = torchaudio.load("clean_trainset_28spk_wav/p226_001.wav")
x = torchaudio.transforms.Resample(sr, 16000)(x)[0]
sr = 16000

# Load RIR
h, _ = torchaudio.load("rir_output_50k/test/wavs/rir_000005.wav")
h = h[0]
h = h

# Forward model
y = conv1d_full(x, h)
y = y

print("Starting CG deconvolution...")
x_rec = conjugate_gradient(y, h, num_iter=80, lambda_reg=1e-3)

# Troncature uniquement ici
x_rec = x_rec[:len(x)]
x_rec = x_rec / (x_rec.abs().max())
x= x / (x.abs().max() )

# Plot
plt.figure(figsize=(12,4))
plt.plot(x.numpy(), label="x original", alpha=0.6)
plt.plot(x_rec.numpy(), label="x_rec (CG corrigé)")
plt.legend()
plt.show()

sd.play(y.numpy(), sr); sd.wait()
sd.play(x_rec.numpy(), sr); sd.wait()
sd.play(x.numpy(), sr); sd.wait()
