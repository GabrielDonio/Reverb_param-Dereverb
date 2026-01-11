import torch 
import torchaudio
from DataRIR import RIRDataset
from Forward import ForwardDiffusion
from DiffWav import DiffWave  
import torch.utils.data as Dataloaders
import glob
import os

# Hyperparameters
batch_size = 16        
num_epochs = 20
learning_rate = 2e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_samples = 10000

# Load dataset
def max_files(files, N):
    nb_files = len(files)
    pick = torch.randperm(nb_files)[:N]
    return [files[i] for i in pick]

NFiles = 5000
FILES = glob.glob("rir_output_50k/train/wavs/*.wav")
selected_files = max_files(FILES, NFiles)
print(f"Found {len(selected_files)} RIR files for training.")

dataset = RIRDataset(selected_files, N=N_samples)
dataloader = Dataloaders.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DiffWave(n_layers=20, res_channels=32, skip_channels=32).to(device)
if os.path.exists("diffwave_rir2_model.pth"):
    model.load_state_dict(torch.load("diffwave_rir2_model.pth", map_location=device,weights_only=True))
    print("Loaded existing model weights.")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
forward_diffusion = ForwardDiffusion(T=200)

scaler = torch.amp.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")

model.train()
print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0

    for batch_idx, (rir_waveforms, _) in enumerate(dataloader):
        if (batch_idx+1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}")
        x0 = rir_waveforms.squeeze(1).to(device)
        t = torch.randint(1, forward_diffusion.T+1, (x0.shape[0],), device=device).long()

        x_t, eps = forward_diffusion.forward(x0, t)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            predicted_noise = model(x_t, t)
            loss = criterion(predicted_noise, eps)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "diffwave_rir2_model.pth")
print("Training complete. Model saved.")
