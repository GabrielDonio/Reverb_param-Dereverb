import torch
import torchaudio
from SimpleDisc import Pix2pixDiscriminator
from TransformModel import ConformerModel
import os
import glob
import matplotlib.pyplot as plt
import sounddevice as sd
import torch.nn.functional as F
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
generator = ConformerModel(freqdim=256, heads=8, ffn_dim=192, depth=3, conv_kernel_size=41, dropout=0.1).to(device)
Crop_LENGTH = 137 
if os.path.exists("LearnVoice/generator1.pth"):
    generator.load_state_dict(torch.load("LearnVoice/generator1.pth", map_location=device, weights_only=True))

generator.eval()
CLEAN_AUDIO_DIR = "clean_testset_wav/"
NOISY_AUDIO_DIR = "noisy_testset_wav/"
CLEAN_AUDIO_DIR = glob.glob(CLEAN_AUDIO_DIR + "*.wav")
NOISY_AUDIO_DIR = glob.glob(NOISY_AUDIO_DIR + "*.wav")
random_index = torch.randint(0, len(CLEAN_AUDIO_DIR), (1,)).item()
clean_path = CLEAN_AUDIO_DIR[random_index]
noisy_path = NOISY_AUDIO_DIR[random_index]
clean_wave, sr = torchaudio.load(clean_path)     
noisy_wave, sr = torchaudio.load(noisy_path)

# resample waveforms and keep them
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
clean_wave = resampler(clean_wave)
noisy_wave = resampler(noisy_wave)

# build spectrograms from the resampled waveforms
N_FFT = 512
HOP_LENGTH = 128
clean_spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=1)(clean_wave)
noisy_spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=1)(noisy_wave)
clean_spec = torch.log1p(clean_spec)
noisy_spec = torch.log1p(noisy_spec)
clean_spec = clean_spec[:,:-1,:]
noisy_spec = noisy_spec[:,:-1,:]
clean_spec = clean_spec.to(device)  
noisy_spec = noisy_spec.to(device)
# handle channel dim safely, permute to [T, F], add batch dim [B=1, T, F]
if clean_spec.dim() == 3 and clean_spec.shape[0] == 1:
    clean_spec = clean_spec.squeeze(0)
if noisy_spec.dim() == 3 and noisy_spec.shape[0] == 1:
    noisy_spec = noisy_spec.squeeze(0)

noisy_spec = noisy_spec.permute(1,0)   # [T, F]
clean_spec = clean_spec.permute(1,0)   # [T, F]

noisy_spec = noisy_spec.unsqueeze(0)   # [1, T, F]
clean_spec = clean_spec.unsqueeze(0)   # [1, T, F]



def plot_spectrogram(spec, title="Spectrogram", ylabel="freq_bin", aspect="auto", xmax=None):

    spec = spec.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.T, origin="lower", aspect=aspect)
    plt.colorbar()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("frame")
    if xmax:
        plt.xlim((0, xmax))
    plt.tight_layout()
def invert_log_mag_spectrogram(
    log_mag_spec,      # [B, T, 256]
    n_fft=512,
    hop_length=128,
    win_length=None,
    n_iter=64,
    device=None
):
    """
    Inversion spectrogramme log-magnitude (power=1) vers waveform
    via Griffin-Lim.
    """
    if device is None:
        device = log_mag_spec.device

    if win_length is None:
        win_length = n_fft

    # -----------------------
    # 1) log → magnitude
    # -----------------------
    mag = torch.expm1(log_mag_spec)
    mag = torch.clamp(mag, min=1e-8)

    # -----------------------
    # 2) remettre la bande manquante (256 → 257)
    #    on duplique la dernière bande (standard et stable)
    # -----------------------
    last_band = mag[:, :, -1:].clone()      # [B, T, 1]
    mag = torch.cat([mag, last_band], dim=-1)  # [B, T, 257]

    # -----------------------
    # 3) format pour Griffin-Lim
    #    [B, F, T]
    # -----------------------
    mag = mag.transpose(1, 2)

    # -----------------------
    # 4) Griffin-Lim
    # -----------------------
    griffinlim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0,
        n_iter=n_iter
    ).to(device)

    wav = griffinlim(mag)   # [B, samples]

    return wav

with torch.no_grad():
    enhanced_spec = generator(noisy_spec) 
    enhanced_spec = enhanced_spec * noisy_spec

    enhanced_wave = invert_log_mag_spectrogram(enhanced_spec, n_fft=N_FFT, hop_length=HOP_LENGTH, device=device)
    enhanced_wave = enhanced_wave.cpu().squeeze().numpy()

    plot_spectrogram(enhanced_spec, title="Enhanced Spectrogram")
    plot_spectrogram(clean_spec, title="Clean Spectrogram")
    plot_spectrogram(noisy_spec, title="Noisy Spectrogram")
    plt.show()
    plt.plot(enhanced_wave, label="Enhanced", alpha=0.7)
    plt.show()
    plt.plot(clean_wave.cpu().squeeze().numpy(), label="Clean", alpha=0.7)
    plt.show()
    plt.plot(noisy_wave.cpu().squeeze().numpy(), label="Noisy", alpha=0.7)
    plt.show()