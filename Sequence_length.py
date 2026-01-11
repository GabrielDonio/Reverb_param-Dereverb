import torchaudio
from torchaudio.transforms import Resample
import glob
import random
import matplotlib.pyplot as plt

Files= glob.glob('reverb_trainset_28spk_wav/*.wav')
RevFiles= glob.glob('clean_trainset_28spk_wav/*.wav')
n=random.randint(0,len(Files)-1)
waveform, sample_rate = torchaudio.load(Files[n])
waveform_clean, sample_rate_clean = torchaudio.load(RevFiles[n])
resampler = Resample(orig_freq=sample_rate, new_freq=22050)
waveform = resampler(waveform)
waveform_clean = resampler(waveform_clean)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(waveform[0, 9000:waveform.shape[1]-1000].numpy())
plt.plot(waveform_clean[0, 9000:waveform_clean.shape[1]-1000].numpy())
plt.title('Waveform Comparison')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(['Reverberant', 'Clean'])
plt.show()
print(waveform.shape)  
print(waveform_clean.shape)