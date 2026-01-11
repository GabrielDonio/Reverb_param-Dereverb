import os
import torch 
import torchaudio
import matplotlib.pyplot as plt
from Diffwave import DiffWave
from Forward import ForwardDiffusion

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 10000 
T_SAMPLES = 200

# Chargement du modèle
model = DiffWave(n_layers=30, res_channels=64, skip_channels=64).to(device)
if os.path.exists("diffwave_rir_model.pth"):
    model.load_state_dict(torch.load("diffwave_rir_model.pth", map_location=device))
    print("✅ Poids du modèle chargés avec succès.")
else:
    print("⚠️ Attention: Fichier 'diffwave_rir_model.pth' introuvable. Utilisation de poids aléatoires.")

model.eval()

forward_diffusion = ForwardDiffusion(T=T_SAMPLES)

@torch.no_grad()
def sample(model, forward_diffusion, N, device):
    T = forward_diffusion.T
    x = torch.randn(1, N).to(device)

    betas = forward_diffusion.beta_t.to(device)
    alphas = forward_diffusion.alpha_t.to(device)
    alpha_cumprod = forward_diffusion.alpha_cumprod.to(device)

    for t in reversed(range(1, T + 1)):
        idx = t - 1
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        eps_theta = model(x, t_tensor)

        mean = (1 / torch.sqrt(alphas[idx])) * (
            x - betas[idx] / torch.sqrt(1 - alpha_cumprod[t]) * eps_theta
        )

        if t > 1:
            sigma_t = torch.sqrt(
                (1 - alpha_cumprod[t-1]) / (1 - alpha_cumprod[t]) * betas[idx]
            )
            x = mean + sigma_t * torch.randn_like(x)
        else:
            x = mean

    return x.squeeze(0).cpu()


generated = sample(model, forward_diffusion, N, device)

generated = generated / (torch.max(torch.abs(generated)) + 1e-7)

# Affichage
plt.figure(figsize=(10, 4))
plt.plot(generated.numpy())
plt.title(f"Signal généré par DiffWave ({T_SAMPLES} étapes)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Sauvegarde optionnelle pour écouter le résultat
# torchaudio.save("generated_rir.wav", generated.unsqueeze(0), 16000)