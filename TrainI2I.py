import torch

import torch
import random

class TrainDerev:
    def __init__(self, model, loss_fn, optimizer, forward, scaler=None, accum_steps=2):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.forward = forward  
        self.N = forward.N      
        self.scaler = scaler
        self.accum_steps = accum_steps
        self._step = 0
    
    def train_step(self, x, y):
        # Convertir TOUJOURS en [B, 1, L] pour l'entraînement
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if y.dim() == 2:
            y = y.unsqueeze(1)
        if x.dim() == 3 and x.size(1) != 1:
            x = x.mean(dim=1, keepdim=True)
        if y.dim() == 3 and y.size(1) != 1:
            y = y.mean(dim=1, keepdim=True)
        
        batch_size = x.shape[0]
        t = torch.rand(batch_size, device=x.device)

        with torch.amp.autocast(device_type=x.device.type, enabled=(self.scaler is not None)):
            # x et y sont [B, 1, L], donc x_t sera [B, 1, L]
            x_t = self.forward.forward(x, y, t)
            
            # Le modèle reçoit [B, 1, L] et devrait retourner [B, 1, L]
            pred = self.model(x_t, t)
            
            # DEBUG: Vérifiez la forme immédiatement
            # print(f"DEBUG - pred shape: {pred.shape}, x_t shape: {x_t.shape}")
            
            # Si pred est 2D ([B, L]), convertissez-le en 3D
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)
                print("ATTENTION: pred était 2D, converti en 3D")
            
            # Replace the loss calculation section with:
            sigma_t = torch.sqrt(self.forward.sigma2_t(t).clamp(min=1e-7))
            sigma_t = sigma_t.view(-1, 1, 1)
            x_target = (x_t - x) / sigma_t

            # Ensure pred and x_target have same shape
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)
            if x_target.dim() == 2:
                x_target = x_target.unsqueeze(1)

            loss_unreduced = self.loss_fn(pred, x_target)
            loss = loss_unreduced #/ self.accum_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._step += 1
        if self._step % self.accum_steps == 0:
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        val = loss_unreduced.item()
        del x_t, pred, loss, loss_unreduced
        return val