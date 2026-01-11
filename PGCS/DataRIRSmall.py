import torchaudio
import torch
import glob
import torch.utils.data as Datasets
import random

class RIRDataset(Datasets.Dataset):
    def __init__(self, rir_file_list, transform=None,N=8000):
        self.rir_file_list = rir_file_list
        self.transform = transform
        self.N = N

    def __len__(self):
        return len(self.rir_file_list)

    def __getitem__(self, idx):
        rir_path = self.rir_file_list[idx]
        rir_waveform, sample_rate = torchaudio.load(rir_path)
        num_channels = rir_waveform.shape[0]
        #channel_idx = random.randint(0, num_channels - 1)
        wavform = rir_waveform[0:1, :] 
        if wavform.shape[1] >= self.N:
            wavform = wavform[:, :self.N]
        else:
            pad = self.N - wavform.shape[1]
            wavform = torch.nn.functional.pad(wavform, (0, pad))
        return wavform, sample_rate
    
if __name__ == "__main__":
    rir_files = glob.glob("rir_output_50k/train/wavs/*.wav")
    dataset = RIRDataset(rir_files)
    dataloader = Datasets.DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_idx, (rir_waveforms, sample_rates) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"RIR Waveforms shape: {rir_waveforms.shape}")
        print(f"Sample Rates: {sample_rates}")
