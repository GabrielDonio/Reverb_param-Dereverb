import torch

class ForwardRev:
    def __init__(self, esp_t=None):
        if esp_t is not None:
            self.esp_t = esp_t
        else:
            self.esp_t = self.epst
    def forward(self, x, y, t):
        n=torch.randn_like(x)
        x_t= (1-t)*x + t*y + t*self.esp_t(t)*n
        return x_t
    def epst(self, t):
        return 0.05/torch.sqrt(t+1e-5)

