import os
import torch
import glob
from Bucketing import DataReverb, collate_pad, ShuffledBatchSampler
from Diffwave import DiffWave
from TrainI2I import TrainDerev
from ForwardRev import ForwardBridge
from LossReverb import LossReverb
import torch.nn as nn

# Hyperparameters
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 200
ACCUM_STEPS = 2  # Gradient accumulation steps

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Chargement des fichiers
clean_files = [r"A:\Program Files\Documents infaudio\Reverb_param\preprocessed\clean\000000.pt"]
reverb_files = [r"A:\Program Files\Documents infaudio\Reverb_param\preprocessed\reverb\000000.pt"]

# Check if files exist
print(f"Clean file exists: {os.path.exists(clean_files[0])}")
print(f"Reverb file exists: {os.path.exists(reverb_files[0])}")

# Load data
clean_data = torch.load(clean_files[0], weights_only=True)
reverb_data = torch.load(reverb_files[0], weights_only=True)
print(f"Clean data shape: {clean_data.shape}")
print(f"Reverb data shape: {reverb_data.shape}")

# Create dataset
dataset = DataReverb(clean_files, reverb_files, N=0)  # Set N=0 to use all data

# Create batch sampler
batches = [
    list(range(i, min(i + BATCH_SIZE, len(dataset))))
    for i in range(0, len(dataset), BATCH_SIZE)
]
batch_sampler = ShuffledBatchSampler(batches, shuffle=False)  # No shuffle for debugging

# Create dataloader
loader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=collate_pad,
)

# Initialize model and components
model = DiffWave().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load checkpoint if exists
if os.path.exists("derev_model.pth"):
    print("Loading existing model...")
    model.load_state_dict(torch.load("derev_model.pth", map_location=device))
else:
    print("Training from scratch...")

# Initialize components
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = LossReverb()
forward = ForwardBridge(device=device)
scaler = torch.amp.GradScaler(device=device.type if device.type == 'cuda' else 'cpu')
trainer = TrainDerev(model, criterion, optimizer, forward, scaler, accum_steps=ACCUM_STEPS)

# Training loop
best_loss = float('inf')
model.train()  # Set to training mode

for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0
    
    for batch_idx, (clean, reverb, lengths) in enumerate(loader):
        # Move to device
        clean = clean.to(device)
        reverb = reverb.to(device)
        
        # Print batch info for debugging
        if batch_idx == 0 and epoch == 0:
            print(f"\nBatch shapes - Clean: {clean.shape}, Reverb: {reverb.shape}")
            print(f"Clean min/max: {clean.min():.4f}/{clean.max():.4f}")
            print(f"Reverb min/max: {reverb.min():.4f}/{reverb.max():.4f}")
        
        # Training step
        loss = trainer.train_step(clean, reverb)
        total_loss += loss
        batch_count += 1
        
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.6f}")
        
        if batch_idx >= 200:  # Limit batches for debugging
            break
    
    # Average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    
    # Update scheduler
    scheduler.step()
    
    print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.8f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_derev_model.pth")
        print(f"  -> Saved new best model with loss: {best_loss:.8f}")

# Save final model
torch.save(model.state_dict(), "final_derev_model.pth")
print(f"\nTraining completed. Best loss: {best_loss:.8f}")

# Test the trained model
print("\n" + "="*50)
print("TESTING THE TRAINED MODEL")
print("="*50)

model.eval()  # Set to evaluation mode

# Test with the same batch used in training
with torch.no_grad():
    test_x = clean[:2].to(device)  # Use first 2 samples
    test_y = reverb[:2].to(device)
    
    # Generate random time steps
    t_test = torch.rand(test_x.shape[0], device=device)
    
    # Forward process
    x_t_test = forward.forward(test_x, test_y, t_test)
    
    # Model prediction
    pred_target_test = model(x_t_test, t_test)
    
    # Calculate true target
    sigma_t_test = torch.sqrt(forward.sigma2_t(t_test).clamp(min=1e-7)).view(-1, 1, 1)
    true_target_test = (x_t_test - test_x) / sigma_t_test
    
    # Adjust dimensions if necessary
    if pred_target_test.dim() == 2:
        pred_target_test = pred_target_test.unsqueeze(1)
    if true_target_test.dim() == 2:
        true_target_test = true_target_test.unsqueeze(1)
    
    # Calculate loss
    test_loss = criterion(pred_target_test, true_target_test)
    print(f"\nTest loss on training sample: {test_loss:.8f}")
    
    # Reconstruction test
    x0_est = x_t_test - sigma_t_test * pred_target_test
    reconstruction_error = torch.mean((x0_est - test_x) ** 2).item()
    print(f"Reconstruction MSE: {reconstruction_error:.8f}")
    print(f"Signal power (test_x): {torch.mean(test_x ** 2).item():.8f}")
    
    # Signal-to-noise ratio
    snr = 10 * torch.log10(torch.mean(test_x ** 2) / torch.mean((x0_est - test_x) ** 2))
    print(f"Reconstruction SNR: {snr.item():.2f} dB")

# Additional debugging: Check model output shapes
print("\n" + "="*50)
print("MODEL SHAPE VERIFICATION")
print("="*50)

# Test different input shapes
test_input_2d = torch.randn(2, 16000, device=device)
test_input_3d = torch.randn(2, 1, 16000, device=device)
test_t = torch.rand(2, device=device)

with torch.no_grad():
    output_2d = model(test_input_2d, test_t)
    output_3d = model(test_input_3d, test_t)
    
    print(f"Input 2D: {test_input_2d.shape} -> Output: {output_2d.shape}")
    print(f"Input 3D: {test_input_3d.shape} -> Output: {output_3d.shape}")
    
    # Verify they're the same
    output_2d_reshaped = output_2d.unsqueeze(1)
    diff = torch.abs(output_2d_reshaped - output_3d).max().item()
    print(f"Max difference between outputs: {diff:.6f}")