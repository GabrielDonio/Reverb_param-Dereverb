import os
import torch
import torchaudio
from tqdm import tqdm
import matplotlib.pyplot as plt
from DiffWav import CausalDereverb
import random
import sounddevice as sd

CH = 32
TIME_DIM = 128
SAVE_DIR = "checkpoints"

def sample_dereverb(model, reverb_audio, steps=50, eps=0.005, device="cuda"):
    model.eval()
    delta = 1.0 / steps
    n = torch.randn_like(reverb_audio)
    x = reverb_audio + eps * n
    with torch.no_grad():
        for i in tqdm(range(steps), desc="Sampling"):
            t_val = 1.0 - i * delta 
            t = torch.tensor([t_val], device=device)

            predicted_clean = model(x, t)

            coeff = delta / t_val
            x_new = coeff * predicted_clean + (1.0 - coeff) * x

            t_next = t_val - delta
            if t_next > 0:
                pass

            x = x_new

    model.train()
    return x

if __name__ == "__main__":
    device = "cuda"
    model = CausalDereverb(ch=CH, time_dim=TIME_DIM, device=device).to(device)
    if os.path.exists(os.path.join(SAVE_DIR, "last.pt")):
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "last.pt"), map_location=device, weights_only=True))
    print("Loaded model for sampling")

    # --- Choix aléatoire d'un fichier ---
    reverb_dir = "reverb_testset_wav"
    clean_dir = "clean_testset_wav"
    all_files = [f for f in os.listdir(reverb_dir) if f.endswith(".wav")]
    chosen_file = random.choice(all_files)
    print(f"Fichier choisi : {chosen_file}")

    reverb_audio, sr_rev = torchaudio.load(os.path.join(reverb_dir, chosen_file))
    clean_audio, sr_clean = torchaudio.load(os.path.join(clean_dir, chosen_file))

    reverb_audio = torchaudio.functional.resample(reverb_audio, orig_freq=sr_rev, new_freq=16000)
    clean_audio = torchaudio.functional.resample(clean_audio, orig_freq=sr_clean, new_freq=16000)

    spec_transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, power=1)

    reverb_audio = spec_transform(reverb_audio)
    clean_audio = spec_transform(clean_audio)

    reverb_audio = torch.log1p(reverb_audio)
    clean_audio = torch.log1p(clean_audio)

    reverb_audio = reverb_audio[:,:-1,:]
    clean_audio = clean_audio[:,:-1,:]

    reverb_audio = reverb_audio.permute(0,2,1).unsqueeze(0).to(device)
    clean_audio = clean_audio.permute(0,2,1).unsqueeze(0).to(device)

    dereverb_audio = sample_dereverb(model, reverb_audio, steps=50, eps=0.001, device=device)

    print(f"L1 Loss: {torch.nn.L1Loss()(dereverb_audio, clean_audio).item()}")

    # --- Inversion spectrogramme → waveform ---
    def invert_log_mag_spectrogram(log_mag_spec, n_fft=512, hop_length=128, n_iter=500, device=None):
        if device is None:
            device = log_mag_spec.device
        mag = log_mag_spec.squeeze(1)
        mag = torch.expm1(mag)
        mag = torch.clamp(mag, min=1e-5)
        last_band = mag[:, :, -1:].clone()
        mag = torch.cat([mag, last_band], dim=-1)
        mag = mag.transpose(1, 2)
        griffinlim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
            power=1.0, n_iter=n_iter
        ).to(device)
        return griffinlim(mag)

    SR = 16000

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

    plot_spectrogram(reverb_audio, title=f"Réverbéré ({chosen_file})", ylabel="freq_bin", aspect="auto")
    plot_spectrogram(dereverb_audio, title=f"Déréverbéré ({chosen_file})", ylabel="freq_bin", aspect="auto")
    plot_spectrogram(clean_audio, title=f"Clean ({chosen_file})", ylabel="freq_bin", aspect="auto")
    plt.show()

    dereverb_wave = invert_log_mag_spectrogram(dereverb_audio, device=device).cpu().squeeze().numpy()
    reverb_wave = invert_log_mag_spectrogram(reverb_audio, device=device).cpu().squeeze().numpy()
    clean_wave = invert_log_mag_spectrogram(clean_audio, device=device).cpu().squeeze().numpy()

        # --- Écoute ---
    print("▶ Lecture : Réverbéré")
    sd.play(reverb_wave, samplerate=SR)
    sd.wait()

    print("▶ Lecture : Déréverbéré")
    sd.play(dereverb_wave, samplerate=SR)
    sd.wait()

    print("▶ Lecture : Clean")
    sd.play(clean_wave, samplerate=SR)
    sd.wait()




