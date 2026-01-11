import torch 
from DataRIR import RIRDataset
from Forward import ForwardDiffusion
from DiffWav import DiffWave  
import torch.utils.data as Dataloaders
import glob
import os

# Hyperparameters
batch_size = 16 
alpha = 2.0       
num_epochs = 20
learning_rate = 2e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_samples = 10000

# Load dataset
def max_files(files, N):
    nb_files = len(files)
    pick = torch.randperm(nb_files)[:N]
    return [files[i] for i in pick]

NFiles = 7000
FILES = glob.glob("rir_output_50k/train/wavs/*.wav")
selected_files = max_files(FILES, NFiles)
print(f"Found {len(selected_files)} RIR files for training.")

dataset = RIRDataset(selected_files, N=N_samples)
dataloader = Dataloaders.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DiffWave(n_layers=20, res_channels=32, skip_channels=32).to(device)
if os.path.exists("diffwave_rirtest_model.pth"):
    model.load_state_dict(torch.load("diffwave_rirtest_model.pth", map_location=device,weights_only=True))
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

    for batch_idx, (rir_waveforms, rir_len) in enumerate(dataloader):
        if (batch_idx+1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}")

        x0 = rir_waveforms.squeeze(1).to(device)   
        rir_len = rir_len.to(device)               

        B, T = x0.shape

        t = torch.randint(
            1, forward_diffusion.T + 1,
            (B,), device=device
        ).long()

        x_t, eps = forward_diffusion.forward(x0, t)

        time_idx = torch.arange(T, device=device).unsqueeze(0)  
        mask = (time_idx < rir_len.unsqueeze(1)).float()        

        time_weight = 1.0 + alpha * (time_idx / T)              

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            predicted_noise = model(x_t, t)

            loss = (predicted_noise - eps) ** 2
            loss = loss * mask * time_weight
            loss = loss.sum() / (mask.sum() + 1e-8)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "diffwave_rirtest_model.pth")
print("Training complete. Model saved.")
