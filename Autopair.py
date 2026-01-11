import random
import matplotlib.pyplot as plt
import torch
import glob

Preproccessed = "preprocessed"

clean_files = sorted(glob.glob(f"{Preproccessed}/clean/*.pt"))
reverb_files = sorted(glob.glob(f"{Preproccessed}/reverb/*.pt"))
lengths = torch.load(f"{Preproccessed}/lengths.pt", weights_only=True)

for i in range(4):
    n = random.randint(0, len(clean_files) - 1)
    clean = torch.load(clean_files[n], weights_only=True)
    reverb = torch.load(reverb_files[n], weights_only=True)
    print(f"Sample {n} - longueur : {lengths[n]}")
    plt.figure(figsize=(12, 4))
    plt.plot(clean.numpy(), label="Clean")
    plt.plot(reverb.numpy(), label="Reverb", alpha=0.5)
    plt.title(f"Sample {n} (longueur : {lengths[n]})")
    plt.legend()
    plt.show()