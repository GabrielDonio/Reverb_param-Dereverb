import torch 
import torchaudio
from DataRIR import RIRDataset
from Diffwave import DiffWave
from Forward import ForwardDiffusion
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
    selected_files = [files[i] for i in pick]
    return selected_files
NFiles = 2000
FILES =  glob.glob("rir_output_50k/train/wavs/*.wav")
selected_files = max_files(FILES, NFiles)
print(f"Found {len(selected_files)} RIR files for training.")
dataset = RIRDataset(selected_files, N=N_samples)
dataloader = Dataloaders.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = DiffWave().to(device)
if os.path.exists("diffwave_rir_model.pth"):
    model.load_state_dict(torch.load("diffwave_rir_model.pth", map_location=device, weights_only=True))
    print("Loaded existing model weights from 'diffwave_rir_model.pth'.")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()
forward_diffusion = ForwardDiffusion(T=200)

model.train() 
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0
    for batch_idx, (rir_waveforms, _) in enumerate(dataloader):
        if (batch_idx+1) % 30 == 0:
            print(f"  Processing batch {batch_idx+1}/{len(dataloader)}")
        x0 = rir_waveforms.squeeze(1).to(device) 

        t = torch.randint(1, forward_diffusion.T+1, (x0.shape[0],), device=device).long()

        x_t, eps = forward_diffusion.forward(x0, t)
        
        optimizer.zero_grad()
        
        predicted_noise = model(x_t, t)
        
        loss = criterion(predicted_noise, eps) 
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.6f}")
torch.save(model.state_dict(), "diffwave_rir_model.pth")
print("Training complete. Model saved as 'diffwave_rir_model.pth'.")