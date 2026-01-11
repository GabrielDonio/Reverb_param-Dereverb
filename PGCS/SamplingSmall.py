import torch
import torchaudio
import matplotlib.pyplot as plt
from DiffWav import DiffWave
from Forward import ForwardDiffusion
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 10000        
T_SAMPLES = 200  

# Charger le modèle
model = DiffWave(n_layers=20, res_channels=32, skip_channels=32).to(device)
if os.path.exists("diffwave_rir2_model.pth"):
    model.load_state_dict(torch.load("diffwave_rir2_model.pth", map_location=device))
    print("✅ Poids chargés avec succès.")
else:
    print("⚠️ Poids non trouvés. Le modèle utilise des poids aléatoires.")

model.eval()

forward_diffusion = ForwardDiffusion(T=T_SAMPLES)

@torch.no_grad()
def sample(model, forward_diffusion, N, device):
    x = torch.randn(1, N).to(device)  
    betas = forward_diffusion.beta_t.to(device)
    alphas = forward_diffusion.alpha_t.to(device)
    alpha_cumprod = forward_diffusion.alpha_cumprod.to(device)

    print("Début du débruitage...")
    for t in reversed(range(1, forward_diffusion.T + 1)):
        idx = t - 1
        t_tensor = torch.full((1,), t, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda"):
            eps_theta = model(x, t_tensor)

        coeff1 = 1 / torch.sqrt(alphas[idx])
        coeff2 = betas[idx] / torch.sqrt(1 - alpha_cumprod[t])
        mean = coeff1 * (x - coeff2 * eps_theta)
        if t > 1:
            alpha_bar_t = alpha_cumprod[t]
            alpha_bar_prev = alpha_cumprod[t-1]
            sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * betas[idx])
            noise = torch.randn_like(x)
            x = mean + sigma_t * noise
        else:
            x = mean

        if t % 50 == 0:
            print(f"Étape {t}/{forward_diffusion.T} complétée...")
    x = x.squeeze(0).cpu()
    return x

# Génération
generated = sample(model, forward_diffusion, N, device)

# Affichage
plt.figure(figsize=(10,4))
plt.plot(generated.numpy())
plt.title(f"Signal généré par DiffWaveSmall ({T_SAMPLES} étapes)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
