import torch
import torchaudio
import matplotlib.pyplot as plt
import glob
import os

CLEAN_AUDIO_DIR = "clean_trainset_56spk_wav/"
CLEAN_AUDIO_DIR = glob.glob(CLEAN_AUDIO_DIR + "*.wav")
NOISY_AUDIO_DIR = "noisy_trainset_56spk_wav/"
NOISY_AUDIO_DIR = glob.glob(NOISY_AUDIO_DIR + "*.wav")
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
i=0
for clean_path, noisy_path in zip(CLEAN_AUDIO_DIR, NOISY_AUDIO_DIR):
    clean, sr = torchaudio.load(clean_path)
    noisy, sr = torchaudio.load(noisy_path)
    clean_spec = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(clean)
    noisy_spec = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(noisy)
    clean_spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH,power=1)(clean_spec)
    clean_spec = torch.log1p(clean_spec)
    noisy_spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH,power=1)(noisy_spec)
    noisy_spec = torch.log1p(noisy_spec)
    clean_spec = clean_spec[:,:-1,:]
    noisy_spec = noisy_spec[:,:-1,:]
    clean_spec = clean_spec.permute(0,2,1)
    noisy_spec = noisy_spec.permute(0,2,1)
    name_clean = os.path.splitext(os.path.basename(clean_path))[0]
    name_noisy = os.path.splitext(os.path.basename(noisy_path))[0]
    os.makedirs("spec/clean_spectrograms", exist_ok=True)
    os.makedirs("spec/noisy_spectrograms", exist_ok=True)
    torch.save(clean_spec, f"spec/clean_spectrograms/{name_clean}.pt")
    torch.save(noisy_spec, f"spec/noisy_spectrograms/{name_noisy}.pt")
    i+=1
    if i % 500 == 0:
        print(f"Processed {i} files")

