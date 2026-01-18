import os
import torch
import glob
from Bucketing import DataReverb, collate_pad, ShuffledBatchSampler
from Diffwave import DiffWave
from TrainI2I import TrainDerev
from ForwardRev import ForwardBridge
from LossReverb import LossReverb
#Hyperparameters
BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 10



# Chargement des fichiers
clean_files = sorted(glob.glob("preprocessed/clean/*.pt"))
reverb_files = sorted(glob.glob("preprocessed/reverb/*.pt"))
lengths = torch.load("preprocessed/lengths.pt", weights_only=True)
sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
files1_sorted = [clean_files[i] for i in sorted_indices]
files2_sorted = [reverb_files[i] for i in sorted_indices]

dataset = DataReverb(files1_sorted, files2_sorted)

batches = [
    list(range(i, min(i + BATCH_SIZE, len(dataset))))
    for i in range(0, len(dataset), BATCH_SIZE)
]
batch_sampler = ShuffledBatchSampler(batches, shuffle=True)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=collate_pad,
)


model = DiffWave()
if os.path.exists("derev_model.pth"):
    model.load_state_dict(torch.load("derev_model.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = LossReverb()
forward = ForwardBridge()
Scaler = torch.amp.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu")
Trainer = TrainDerev(model, criterion, optimizer, forward, Scaler)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    for batch_idx, (clean, reverb, lengths) in enumerate(loader):
        x = clean.to(device)
        y = reverb.to(device)
        loss = Trainer.train_step(x, y)
        total_loss += loss
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx} of {len(loader)}, Loss: {loss:.4f}")
        if batch_idx>200:
            break
    avg_loss = total_loss / 200
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
torch.save(model.state_dict(), "derev_model.pth")

