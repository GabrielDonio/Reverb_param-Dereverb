import torch
import torch.fft
import torchaudio
from DiffWav import DiffWave
import matplotlib.pyplot as plt


# -------------------------------
# Fonction Wiener
# -------------------------------

def wiener_update(y, h, eps=1e-6):
    """
    Met à jour le son sec x via Wiener (MAP).
    """
    Y = torch.fft.fft(y)
    H = torch.fft.fft(h, n=y.shape[0])
    
    X = torch.conj(H) / (torch.abs(H)**2 + eps) * Y
    x_est = torch.fft.ifft(X).real
    return x_est

# -------------------------------
# Echantillonnage h_t via modèle de diffusion
# -------------------------------

def sample_ht(model, h_next, t):
    """
    Échantillonne h_t depuis h_{t+1} avec DiffWave.
    model : ton modèle de diffusion
    h_next : h_{t+1} ou h_{t, m-1}
    t : pas de diffusion
    """
    # Ensure t is a tensor and on the same device as h_next
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=h_next.device)
    else:
        t = t.to(h_next.device)
    ht_pred = model(h_next, t)
    noise = 0.01 * torch.randn_like(ht_pred)
    return ht_pred + noise

# -------------------------------
# Algorithme PCGS transposé
# -------------------------------

def pcgs_transposed(model, y, x_init, N=3, T=10, M_t=2):
    """
    y : signal observé (1D tensor)
    x_init : initialisation du son sec (1D tensor)
    N : nombre de cycles externes
    T : nombre de pas de reverse diffusion
    M_t : nombre de raffinements internes par pas t
    """
    x_prev = x_init.clone()
    K = 0  # compteur d'updates internes
    h_final = None
    
    for n in range(1, N+1):
        x_n = x_prev.clone()
        K = 0
        
        # Initialisation h_T (on peut partir de y bruité)
        h_t = y.clone() + 0.01 * torch.randn_like(y)
        h_t = sample_ht(model, h_t, t=T)
        
        # Boucle reverse diffusion
        for t in reversed(range(T)):
            h_t = sample_ht(model, h_t, t=t)
            
            # Raffinement interne sur x
            for m in range(1, M_t+1):
                x_n = wiener_update(y, h_t)
                K += 1
                h_t = sample_ht(model, h_t, t=t)
        
        # Stocker les valeurs finales
        x_prev = x_n.clone()
        h_final = h_t.clone()
    
    return h_final, x_prev
if __name__ == "__main__":
    # Chargement y
    y, sr = torchaudio.load("reverberated_noisy.wav")
    print(f"Sample rate: {sr}, Shape of y: {y.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = y.to(device)
    model=DiffWave().to(device)
    model.load_state_dict(torch.load("diffwave_rirtest_model.pth", map_location=device))
    
    # Initialisation x (phi) : on peut commencer par y ou un silence
    x_init = torch.randn_like(y) * 0.01 
    
    h_final, x_final = pcgs_transposed(model, y, x_init, N=1, T=200, M_t=2)
    
    # Sauvegarde
    torchaudio.save("x_restored_pcgs.wav", x_final.cpu().view(1, -1), 16000)
    plt.figure(figsize=(12, 6))
    plt.plot(x_final.cpu().view(-1).numpy())
    plt.show()