import torchaudio
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt

FILES = glob.glob("rir_output_50k/train/wavs/*.wav")
n = torch.randint(0, len(FILES), (1,))
file,sr=torchaudio.load(FILES[n])
plt.plot(file[0].t().numpy())
plt.title("RIR Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude") 
plt.show()