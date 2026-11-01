import torch
import torchaudio
import glob
import os


REVERB_AUDIO_DIR = "reverb_trainset_56spk_wav/"
REVERB_AUDIO_DIR = glob.glob(REVERB_AUDIO_DIR + "*.wav")
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
i=0
for reverb_path in REVERB_AUDIO_DIR:
    reverb, sr = torchaudio.load(reverb_path)

    reverb_spec = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(reverb)

    reverb_spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH,power=1)(reverb_spec)
    reverb_spec = torch.log1p(reverb_spec)

    reverb_spec = reverb_spec[:,:-1,:]

    reverb_spec = reverb_spec.permute(0,2,1)
    name_reverb = os.path.splitext(os.path.basename(reverb_path))[0]

    os.makedirs("spec/reverb_spectrograms", exist_ok=True)

    torch.save(reverb_spec, f"spec/reverb_spectrograms/{name_reverb}.pt")
    i+=1
    if i % 500 == 0:
        print(f"Processed {i} files")