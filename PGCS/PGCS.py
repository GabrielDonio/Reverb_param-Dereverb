import torch 
from DiffWav import DiffWave
from DeconvWiener import wiener_deconvolution_torch as wiener_deconvolution
from Forward import ForwardDiffusion
import torchaudio
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiffWave().to(device)
if os.path.exists("diffwave_rirtest_model.pth"):
    model.load_state_dict(torch.load("diffwave_rirtest_model.pth", map_location=device))
    print("Modèle chargé avec succès.")
model.eval()

def estimate_x(y, h, noise_variance=0.000001):
    return wiener_deconvolution(y.squeeze(), h.squeeze(), noise_variance)

