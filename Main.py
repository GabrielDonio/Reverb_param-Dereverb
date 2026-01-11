import torch
import glob
from Bucketing import DataReverb, collate_pad, ShuffledBatchSampler
from Diffwave import DiffWave
from TrainDerev import TrainDerev
from Forward import ForwardRev
#Hyperparameters
BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 2



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
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()
forward = ForwardRev()
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
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx} of {len(loader)}, Loss: {loss:.4f}")
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

