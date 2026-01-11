import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt

def conv1d_full(x, h):
    """ Convolution qui préserve la queue de réverbération """
    Lx = x.shape[-1]
    Lh = h.shape[-1]
    # On ajoute du padding pour ne pas perdre la fin du son
    y = F.conv1d(x.view(1,1,-1), h.view(1,1,-1), padding=Lh-1)
    return y.squeeze()[:Lx] # On recoupe à Lx pour la comparaison

# 1. Chargement et préparation
x, sr = torchaudio.load("clean_trainset_28spk_wav/p226_001.wav")
x = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(x)
sr = 16000
x = x[0]

h, _ = torchaudio.load("rir_output_50k/train/wavs/rir_000012.wav")
h = h[0]
# Normalisation par l'énergie (plus stable que la somme simple)
h = h / torch.norm(h)

# 2. Création du signal réverbéré (cible)
with torch.no_grad():
    y = conv1d_full(x, h)

# 3. Optimisation
# On initialise avec un peu de y pour aider l'optimiseur au début
x_est = y.clone().detach().requires_grad_(True)

# Un LR plus faible (0.01) évite les bruits de sifflement
optimizer = torch.optim.Adam([x_est], lr=0.01)
num_iter = 200  # Augmenté pour une convergence propre
lambda_reg = 1e-6 # Très faible pour ne pas étouffer la voix

print("Début de la déconvolution...")
for i in range(num_iter):
    optimizer.zero_grad()

    y_hat = conv1d_full(x_est, h)
    
    # Perte principale
    data_loss = F.mse_loss(y_hat, y)

    # Régularisation TV (Total Variation) : élimine le côté métallique
    # On pénalise les variations brusques de manière linéaire (L1)
    dx = x_est[1:] - x_est[:-1]
    reg_loss = lambda_reg * torch.abs(dx).mean()

    loss = data_loss + reg_loss
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        print(f"Iter {i}, Loss: {loss.item():.8f}")

# 4. Post-traitement et Normalisation
x_rec = x_est.detach()
x_rec = x_rec / (x_rec.abs().max() + 1e-8)
x_orig = x / (x.abs().max() + 1e-8)
y_plot = y / (y.abs().max() + 1e-8)

# 5. Visualisation
plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.plot(y_plot.numpy(), label="Réverbéré (y)", alpha=0.5)
plt.plot(x_orig.numpy(), label="Original (x)", alpha=0.5)
plt.legend()
plt.subplot(2,1,2)
plt.plot(x_rec.numpy(), label="Retrouvé (x_rec)", color="green")
plt.legend()
plt.show()
# 6. Écoute
print("Écoute du signal réverbéré...")
sf.write("reverb_signal.wav", y.numpy(), sr)
print("Écoute du signal reconstruit...")
sf.write("reconstructed_signal.wav", x_rec.numpy(), sr)
print("Écoute du signal original...")
sf.write("original_signal.wav", x_orig.numpy(), sr)