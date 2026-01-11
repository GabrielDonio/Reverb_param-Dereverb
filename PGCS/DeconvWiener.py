import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
import torch.fft as fft
import matplotlib.pyplot as plt

def wiener_deconvolution(y, h, noise_variance=0.000001, k=10):
    """
    y : signal réverbéré
    h : RIR
    noise_variance : variance du bruit
    k : facteur d'ajustement (augmente k si ça siffle encore)
    """
    n = len(y) + len(h) - 1
    Y = np.fft.fft(y, n)
    H = np.fft.fft(h, n)
    
    H_conj = np.conj(H)
    H_sq_mag = np.abs(H)**2
    
    reg = k * noise_variance
    
    G = H_conj / (H_sq_mag + reg)
    
    X_hat = Y * G
    x_recovered = np.fft.ifft(X_hat)
    
    result = np.real(x_recovered[:len(y)])
    
    if np.max(np.abs(result)) > 0:
        result = result / np.max(np.abs(result))
        
    return result
def wiener_deconvolution_torch(y, h, noise_variance=1e-6, k=10):
    """
    Version PyTorch de ta fonction Wiener.
    y : [L] (Tensor)
    h : [10000] (Tensor)
    """
    n = y.shape[-1] + h.shape[-1] - 1
    
    Y = fft.rfft(y, n=n)
    H = fft.rfft(h, n=n)
    
    H_conj = torch.conj(H)
    H_sq_mag = torch.abs(H)**2
    
    reg = k * noise_variance
    G = H_conj / (H_sq_mag + reg)
    
    X_hat = Y * G
    
    x_recovered = fft.irfft(X_hat, n=n)

    result = x_recovered[..., :y.shape[-1]]
    
    if torch.max(torch.abs(result)) > 0:
        result = result / torch.max(torch.abs(result))
        
    return result

x, fs = sf.read('clean_trainset_28spk_wav/p226_001.wav')
x= signal.resample(x, int(len(x)*16000/fs))
fs = 16000
h, fs_h = sf.read('rir_output_50k/train/wavs/rir_000012.wav')

if len(x.shape) > 1: x = x[:, 0]
if len(h.shape) > 1: h = h[:, 0]

y_clean = signal.convolve(x, h, mode='full')

var_bruit = 0.00
bruit = np.random.normal(0, np.sqrt(var_bruit), len(y_clean))
y_noisy = y_clean + bruit

# 4. Déconvolution de Wiener
x_recovered = wiener_deconvolution(y_noisy, h, var_bruit)
x_recovered = x_recovered / (np.max(np.abs(x_recovered)) + 1e-8)
x = x / (np.max(np.abs(x)) + 1e-8)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(y_noisy)
plt.title('Signal Réverbéré et Bruité')
plt.xlabel('Échantillons')
plt.ylabel('Amplitude')
plt.show()

# 5. Sauvegarde des résultats
sf.write('reverberated_noisy.wav', y_noisy, fs)
sf.write('recovered_signal.wav', x_recovered, fs)
sf.write('original_signal.wav', x, fs)

print("Traitement terminé. Fichiers sauvegardés.")