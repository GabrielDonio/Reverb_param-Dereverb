import torch
import random

class TrainDerev:
    def __init__(self, model, loss_fn, optimizer, forward, scaler=None, accum_steps=16):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.forward = forward
        self.scaler = scaler
        self.accum_steps = accum_steps
        self._step = 0
    
    def train_step(self, x, y):
        batch_size = x.shape[0]
        t = torch.rand(batch_size, device=x.device, dtype=x.dtype) 
        t_b = t.unsqueeze(-1)

        with torch.amp.autocast(device_type=x.device.type, enabled=(self.scaler is not None)):
            x_t = self.forward.forward(x, y, t_b)
            if x_t.dim() == 2:
                x_t = x_t.unsqueeze(1)
            pred = self.model(x_t, t) 
            loss_unreduced = self.loss_fn(pred, x)
            loss = loss_unreduced / self.accum_steps

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
        torch.cuda.empty_cache()
        return val

