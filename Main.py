import os
import torch
import glob

from DiffWav import AudioWavUnet
from DataReverb import PairedAudioDataset
from torch.utils.data import DataLoader

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 5
LOG_INTERVAL = 20
ALPHA = 0
EPS_NOISE = 0.005

# Chargement des fichiers
clean_files = sorted(glob.glob("preprocessed/clean/*.pt"))
reverb_files = sorted(glob.glob("preprocessed/reverb/*.pt"))
if __name__ == "__main__":
    model = AudioWavUnet()
    if os.path.exists("dereverb_model.pth"):
        model.load_state_dict(torch.load("dereverb_model.pth"))
    device = 'cuda'
    model.to(device)

    dataset = PairedAudioDataset("preprocessed/clean", "preprocessed/reverb",N_min=16000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,fused=True)
    """def loss_fn(output, clean):
        output_fft = torch.fft.rfft(output.squeeze(1).float())
        clean_fft = torch.fft.rfft(clean.squeeze(1).float())
        output_mag = torch.abs(output_fft)
        clean_mag = torch.abs(clean_fft)
        return ALPHA*torch.nn.functional.l1_loss(output_mag, clean_mag) + torch.nn.functional.mse_loss(output, clean)
"""
    loss_fn = torch.nn.MSELoss()

    scaler = torch.amp.GradScaler()

    torch.backends.cudnn.benchmark = True
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_count = 0
        for batch_idx, (reverb, clean) in enumerate(loader):
            reverb, clean = reverb.to(device), clean.to(device)
            reverb = reverb.unsqueeze(1) 
            clean = clean.unsqueeze(1)   
            t = torch.rand(reverb.size(0),device=device) 
            with torch.amp.autocast(device_type=device):
                fct = t[:,None,None]
                reverb = (1-fct)*clean + fct*reverb + EPS_NOISE*torch.randn_like(reverb)
                output = model(reverb, t)
                loss = loss_fn(output, clean)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_count += 1

            if (running_count % LOG_INTERVAL == 0):
                avg = running_loss / running_count
                print(f"Epoch {epoch}  batch {batch_idx}  avg_loss/{LOG_INTERVAL} = {avg:.6f}")
                running_loss = 0.0
                running_count = 0
    torch.save(model.state_dict(), "dereverb_model.pth")