import torch
import os
from Diffwave import DiffWave
from ForwardRev import ForwardBridge
from matplotlib import pyplot as plt

class SamplerDerev:
    def __init__(self, model, forward, device=None, num_steps=50):
        """
        model: réseau εθ (conditionné par le temps continu t ∈ [0,1])
        forward: instance de ForwardBridge (temps continu)
        num_steps: nombre d'étapes de sampling (peut être différent de forward.N)
        """
        self.model = model
        self.forward = forward
        self.num_steps = num_steps
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def sample(self, y):
        """
        y: audio réverbéré X1 [B, 1, T] ou [B, T]
        retourne x̂0: audio clean estimé
        Suit l'Algorithme 2 du papier I2SB, en temps continu.
        """
        self.model.eval()
        if y.dim() == 2:
            y = y.unsqueeze(1)
        x_t = y.to(self.device)  # On part de X1 (t = 1.0)

        # Créer une séquence de temps continus décroissante de 1.0 à ~0.
        times = torch.linspace(1.0, 0.0, self.num_steps + 1, device=self.device)  # [num_steps+1]
        # times[0] = 1.0, times[num_steps] = 0.0
        for i in range(self.num_steps):
            t_current = times[i]      # temps courant (ex: 1.0, 0.98, ...)
            t_prev = times[i + 1]     # temps suivant (plus proche de 0)

            batch_size = x_t.shape[0]
            # Tenseur de temps continu pour tout le batch (forme [B])
            t_continuous = torch.full((batch_size,), t_current, device=self.device, dtype=torch.float32)

            # 1. Prédiction du réseau (identique à l'entraînement)
            eps_pred = self.model(x_t, t_continuous)  # ϵ(Xt, t; θ)
            # S'assurer que eps_pred a la forme [B, 1, L] pour les opérations
            if eps_pred.dim() == 2:
                eps_pred = eps_pred.unsqueeze(1)

            # 2. Estimation de X0 à partir de la prédiction
            sigma_t = torch.sqrt(self.forward.sigma2_t(t_continuous).clamp(min=1e-7)).view(-1, 1, 1)
            x0_est = x_t - sigma_t * eps_pred

            # 3. Calcul des paramètres pour le reverse step (DDPM Eq. 4 adapté au bridge)
            #    Besoin de σ_t² et σ_{t_prev}²
            sigma2_t = self.forward.sigma2_t(t_continuous).clamp(min=1e-7)
            t_prev_tensor = torch.full((batch_size,), t_prev, device=self.device, dtype=torch.float32)
            sigma2_t_prev = self.forward.sigma2_t(t_prev_tensor).clamp(min=1e-7)

            # Coefficients pour la moyenne μ
            coef_xt = sigma2_t_prev / sigma2_t
            coef_x0 = (sigma2_t - sigma2_t_prev) / sigma2_t

            coef_xt = coef_xt.view(-1, 1, 1)
            coef_x0 = coef_x0.view(-1, 1, 1)

            mu = coef_xt * x_t + coef_x0 * x0_est

            # Variance
            var = (sigma2_t_prev * (sigma2_t - sigma2_t_prev)) / sigma2_t
            std = torch.sqrt(var.clamp(min=1e-7)).view(-1, 1, 1)

            # 4. Sampling (ajout de bruit sauf à la dernière étape)
            if i < self.num_steps - 1:
                noise = torch.randn_like(x_t)
            else:
                noise = 0.0  # Dernière étape déterministe (quand t_prev ≈ 0)
            x_t = mu + std * noise

        # Après la boucle, x_t correspond à X0 estimé (à t ≈ 0)
        return x_t.squeeze(1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DiffWave().to(device)
    if os.path.exists("derev_model.pth"):
        model.load_state_dict(torch.load("derev_model.pth", map_location=device))
        print("Model loaded.")

    # NOTE : ForwardBridge doit être sur le MÊME device que le modèle
    forward = ForwardBridge(N=1000, beta_max=10.0, device=device)
    sampler = SamplerDerev(model, forward, device=device, num_steps=50)

    clean_file = r"A:\Program Files\Documents infaudio\Reverb_param\preprocessed\clean\000000.pt"
    reverb_file = r"A:\Program Files\Documents infaudio\Reverb_param\preprocessed\reverb\000000.pt"
    
    clean = torch.load(clean_file, weights_only=True).unsqueeze(0)  # [1, T]
    reverb = torch.load(reverb_file, weights_only=True).unsqueeze(0)  # [1, T]

    # Déplacer les données sur le device
    reverb = reverb.to(device)
    clean = clean.to(device)

    print(f"Clean shape: {clean.shape}, Reverb shape: {reverb.shape}")

    x0 = sampler.sample(reverb)
    print(f"Sampled shape: {x0.shape}")

    l1_loss = torch.nn.L1Loss()(x0, clean)
    print(f"L1 Loss between sampled clean and ground truth clean: {l1_loss.item():.6f}")

    # Affichage
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.title("Clean Audio")
    plt.plot(clean.squeeze().cpu().numpy())
    plt.subplot(3, 1, 2)
    plt.title("Reverberated Audio")
    plt.plot(reverb.squeeze().cpu().numpy())
    plt.subplot(3, 1, 3)
    plt.title("Sampled Clean Audio")
    plt.plot(x0.squeeze().cpu().numpy())
    plt.tight_layout()
    plt.show()