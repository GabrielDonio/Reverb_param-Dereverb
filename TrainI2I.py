import torch
import torch.nn as nn
from torch.utils.data import DataLoader  
from tqdm import tqdm
import os
import copy

from DiffWav import CausalDereverb
from DataReverb import SpectrogramDataset

# --- Config ---
EPOCHS = 6
BATCH_SIZE = 25
LR = 1e-4
CH = 32
TIME_DIM = 128
N = 137
NB_TRAIN = 12000
SAVE_DIR = "checkpoints"
EPS_NOISE = 0.001
ALPHA = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(SAVE_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if __name__ == "__main__":
    # --- Data ---
    train_dataset = SpectrogramDataset("spec/clean_spectrograms/", "spec/reverb_spectrograms/", N=N, Nb_files=NB_TRAIN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # --- Modèle ---
    model = CausalDereverb(ch=CH, time_dim=TIME_DIM, device=device).to(device)
    if os.path.exists(os.path.join(SAVE_DIR, "last.pt")):
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "last.pt"), map_location=device))
   # ema_model = copy.deepcopy(model).to(device)
    #ema_model.eval()
    #ema = EMA(beta=0.995)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse = nn.MSELoss() 
    l1loss = nn.L1Loss()
    scaler = torch.amp.GradScaler()

    print(f"Paramètres: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for reverb, clean in tqdm(train_loader, desc=f"Epoch {epoch}"):
            reverb = reverb.float().unsqueeze(1).to(device)
            clean = clean.float().unsqueeze(1).to(device)
            t = torch.rand(clean.shape[0], device=device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device):
                fct = t[:, None, None, None]
                n = torch.randn_like(reverb)
                x = (1 - fct) * clean + fct * reverb + fct * EPS_NOISE * n   
                pred = model(x , t)
                loss = mse(pred, clean)*ALPHA + l1loss(pred, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * reverb.size(0)

        train_loss /= len(train_dataset)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last.pt"))
        #torch.save(ema_model.state_dict(), os.path.join(SAVE_DIR, "last_ema.pt"))