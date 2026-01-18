import torch
import torch.nn as nn
import torch.nn.functional as F

class LossReverb:
    def __init__(self, lambda_stft=0.5, n_fft=512):
        self.lambda_stft = lambda_stft
        self.n_fft = n_fft
        self.hop_length = n_fft // 4  
        self.win_length = n_fft
        self.mse_loss = nn.MSELoss()

    def lossSTFT(self, pred, target):
        # support tensors with shape [B, C, T] or [B, T]
        if pred.dim() == 3:
            p = pred.reshape(-1, pred.size(-1))   # (B*C, T)
        else:
            p = pred
        if target.dim() == 3:
            t = target.reshape(-1, target.size(-1))
        else:
            t = target
        
        window = torch.hann_window(self.win_length).to(p.device)

        s_pred = torch.stft(p, self.n_fft, self.hop_length, self.win_length, window, return_complex=True).abs()
        s_target = torch.stft(t, self.n_fft, self.hop_length, self.win_length, window, return_complex=True).abs()

        sc_loss = torch.norm(s_target - s_pred, p="fro") / (torch.norm(s_target, p="fro") + 1e-8)
    
        mag_loss = F.l1_loss(torch.log(s_pred + 1e-5), torch.log(s_target + 1e-5))
        
        return sc_loss + mag_loss

    def __call__(self, pred, target):
        mse = self.mse_loss(pred, target)
        #stft = self.lossSTFT(pred, target)
        return mse # + self.lambda_stft * stft