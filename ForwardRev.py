import torch

class ForwardBridge:
    def __init__(self, N=1000, beta_max=10.0, device="cpu"):
        """
        N: nombre de points discrets pour approximer les intégrales.
        beta_max: détermine l'échelle du bruit (β(t) = beta_max * t).
        """
        self.N = N
        self.device = torch.device(device)
        self.beta_max = beta_max

        self.ts = torch.linspace(0, 1, N, device=self.device)  # [N]

        self.betas = self.beta_max * self.ts  

        dt = 1.0 / (N - 1) if N > 1 else 1.0
        self.sigma2 = torch.cumsum(self.betas, dim=0) * dt  # [N] 
        sigma2_1 = self.sigma2[-1]
        self.bar_sigma2 = sigma2_1 - self.sigma2  # [N]

    def _interp(self, t_continuous, precomputed_vals):
        """Interpole linéairement les valeurs pré-calculées pour un temps continu t."""
        # clamp t to valid range and move precomputed values to the same device
        t = t_continuous.clamp(0.0, 1.0)
        if precomputed_vals.device != t.device:
            precomputed_vals = precomputed_vals.to(t.device)

        idx = t * (self.N - 1)
        idx_low = torch.floor(idx).long()
        idx_high = torch.clamp(idx_low + 1, max=self.N - 1)
        weight = idx - idx_low.float()

        val_low = precomputed_vals[idx_low]
        val_high = precomputed_vals[idx_high]
        return val_low * (1 - weight) + val_high * weight

    def sigma2_t(self, t_continuous):
        """Retourne σ²(t) pour un temps continu t (peut être un batch)."""
        return self._interp(t_continuous, self.sigma2)

    def bar_sigma2_t(self, t_continuous):
        """Retourne σ̄²(t) pour un temps continu t (peut être un batch)."""
        return self._interp(t_continuous, self.bar_sigma2)

    def forward(self, X0, X1, t_continuous):
        """
        Échantillonne Xt ~ q(Xt | X0, X1).
        X0, X1: tenseurs de forme [B, C, L].
        t_continuous: temps continus ∈ [0,1] pour chaque élément du batch, shape [B].
        """
        sigma2_t = self.sigma2_t(t_continuous)  
        bar_sigma2_t = self.bar_sigma2_t(t_continuous) 

        view_shape = (-1, *([1] * (X0.dim() - 1)))  
        sigma2_t_v = sigma2_t.view(view_shape)
        bar_sigma2_t_v = bar_sigma2_t.view(view_shape)

        mu_t = (bar_sigma2_t_v / (bar_sigma2_t_v + sigma2_t_v)) * X0 \
               + (sigma2_t_v / (bar_sigma2_t_v + sigma2_t_v)) * X1
        Sigma_t = (sigma2_t_v * bar_sigma2_t_v) / (sigma2_t_v + bar_sigma2_t_v)

        X_t = mu_t + torch.sqrt(Sigma_t) * torch.randn_like(X0)
        return X_t