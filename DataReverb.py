from random import shuffle
import torch
import glob
import torch.utils.data as data

class SpectrogramDataset(data.Dataset):
    def __init__(self, clean_dir, noisy_dir, N, Nb_files=16000):
        clean_files = sorted(glob.glob(clean_dir + "*.pt"))
        noisy_files = sorted(glob.glob(noisy_dir + "*.pt"))
        
        total_files = len(clean_files)        
        indices = torch.randperm(total_files)[:Nb_files]
        self.clean_files = [clean_files[i] for i in indices]
        self.noisy_files = [noisy_files[i] for i in indices]
        
        self.N = N

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_spec = torch.load(self.clean_files[idx], weights_only=True)
        if clean_spec.dim() == 3:
            clean_spec = clean_spec.mean(0) if clean_spec.shape[0] > 1 else clean_spec.squeeze(0)

        noisy_spec = torch.load(self.noisy_files[idx], weights_only=True)
        if noisy_spec.dim() == 3:
            noisy_spec = noisy_spec.mean(0) if noisy_spec.shape[0] > 1 else noisy_spec.squeeze(0)

        crop_time = torch.randint(0, clean_spec.shape[0] - self.N + 1, (1,)).item()
        clean_spec = clean_spec[crop_time:crop_time + self.N]
        noisy_spec = noisy_spec[crop_time:crop_time + self.N]

        return noisy_spec, clean_spec