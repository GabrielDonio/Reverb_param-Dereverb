from random import shuffle
import torch
import glob
import torch.utils.data as data

class SpectrogramDataset(data.Dataset):
    def __init__(self, clean_dir, noisy_dir,N,Nb_files=16000):
        self.clean_files = sorted(glob.glob(clean_dir + "*.pt"))
        self.noisy_files = sorted(glob.glob(noisy_dir + "*.pt"))
        total_files = len(self.clean_files)
        index_start = torch.randint(0, total_files - Nb_files,(1,)).item()
        self.clean_files = self.clean_files[index_start:index_start + Nb_files]
        self.noisy_files = self.noisy_files[index_start:index_start + Nb_files]
    
        self.N = N
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_spec = torch.load(self.clean_files[idx], weights_only=True)  # [C, T, F] ou [T, F]
        if clean_spec.dim() == 3:
            if clean_spec.shape[0] > 1:
                clean_spec = clean_spec.mean(dim=0) 
            else:
                clean_spec = clean_spec.squeeze(0)  
        noisy_spec = torch.load(self.noisy_files[idx], weights_only=True)
        if noisy_spec.dim() == 3:
            if noisy_spec.shape[0] > 1:
                noisy_spec = noisy_spec.mean(dim=0)
            else:
                noisy_spec = noisy_spec.squeeze(0)
        crop_time = torch.randint(0, clean_spec.shape[0] - self.N + 1, (1,)).item()
        clean_spec = clean_spec[crop_time:crop_time + self.N] 
        noisy_spec = noisy_spec[crop_time:crop_time + self.N]
        return noisy_spec, clean_spec
    
if __name__ == "__main__":
    MIN_LENGTH = 137
    """def min_length(parallel=True, workers=8):
        files = sorted(glob.glob("spec/clean_spectrograms/*.pt"))
        if not files:
            return 0
        def get_len(path):
            spec = torch.load(path, weights_only=True, map_location="cpu")
            if spec.dim() == 3:
                if spec.shape[0] > 1:
                    spec = spec.mean(dim=0)
                else:
                    spec = spec.squeeze(0)
            return int(spec.shape[0])
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(workers, len(files))) as ex:
                lengths = list(ex.map(get_len, files))
        else:
            lengths = [get_len(f) for f in files]
        return min(lengths)

"""
    dataset = SpectrogramDataset("spec/clean_spectrograms/", "spec/noisy_spectrograms/",N=MIN_LENGTH)
    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
    for noisy_spec, clean_spec in dataloader:
        print(noisy_spec.shape)
        print(clean_spec.shape)
        break
    
