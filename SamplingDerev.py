import torch

class SamplingDerev:
    def __init__(self, model, forward,delta=0.02, esp_t=None):
        self.model = model
        self.forward = forward
        self.delta = delta
        if esp_t is not None:
            self.esp_t = esp_t
        else:
            self.esp_t = self.default_esp_t

    def default_esp_t(self, t):
        return 1 / torch.sqrt(t)

    @torch.no_grad()
    def sample(self, y):
        device = y.device
        dtype = y.dtype
        n=torch.randn_like(y)
        xt = y + self.esp_t(1)*n
        for t in torch.arange(1 - self.delta, 0, -self.delta, device=device, dtype=dtype):
            n = torch.randn_like(y)
            xt = (self.delta/t)*self.model(xt, t) + (1-self.delta/t)*xt 
            n=(t-self.delta)*torch.sqrt(self.esp_t(t-self.delta)**2-self.esp_t(t)**2)*n
            xt = xt + n
        return xt