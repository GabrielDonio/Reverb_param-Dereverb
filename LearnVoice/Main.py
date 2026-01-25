import torch
import torch.utils.data as data
import torch.optim as optim
import os
import random

from SpectrogramDataset import SpectrogramDataset
from Pix2pixDisc import MediumLightDiscriminator2D 
from TransformModel import ConformerModel

# --- Hyperparamètres ---
MIN_LENGTH = 137
BATCH_SIZE = 16
LR_GEN = 5e-4
LR_DISC = 5e-5       
L1_LAMBDA = 100      
EPOCHS = 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("LearnVoice", exist_ok=True)

    dataset = SpectrogramDataset("spec/clean_spectrograms/", "spec/noisy_spectrograms/", N=MIN_LENGTH, Nb_files=30000)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    generator = ConformerModel(freqdim=256, heads=8, ffn_dim=192, depth=3).to(device)
    discriminator = MediumLightDiscriminator2D().to(device)

    gen_optimizer = optim.Adam(generator.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LR_DISC, betas=(0.5, 0.999))

    criterion_L1 = torch.nn.L1Loss()
    criterion_GAN = torch.nn.MSELoss() 
    
    scaler = torch.amp.GradScaler()

    for epoch in range(EPOCHS):
        running_D, running_G_GAN, running_L1 = 0.0, 0.0, 0.0
        
        for i, (noisy_spec, clean_spec) in enumerate(dataloader):
            noisy_spec, clean_spec = noisy_spec.to(device), clean_spec.to(device)

            # --- ENTRAÎNEMENT DU DISCRIMINATEUR ---
            with torch.amp.autocast(device_type="cuda"):
                fake_spec = generator(noisy_spec) 
                fake_spec = fake_spec*noisy_spec  
                
                d_real = discriminator(clean_spec, noisy_spec)
                d_fake = discriminator(fake_spec.detach(), noisy_spec)
                
                # Injection de bruit dans les labels (0.7-1.2 pour Real, 0.0-0.3 pour Fake)
                # C'est ce qui empêche D de descendre à zéro et de tuer G
                real_label_val = 0.8 + (random.random() * 0.2)
                fake_label_val = random.random() * 0.2
                
                loss_D_real = criterion_GAN(d_real, torch.ones_like(d_real) * real_label_val)
                loss_D_fake = criterion_GAN(d_fake, torch.zeros_like(d_fake) + fake_label_val)
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

            disc_optimizer.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(disc_optimizer)

            # --- ENTRAÎNEMENT DU GÉNÉRATEUR ---
            with torch.amp.autocast(device_type="cuda"):
                d_fake_for_G = discriminator(fake_spec, noisy_spec)
                
                loss_GAN = criterion_GAN(d_fake_for_G, torch.ones_like(d_fake_for_G))
                loss_L1 = criterion_L1(fake_spec, clean_spec)
                
                loss_G = loss_GAN + (L1_LAMBDA * loss_L1)

            gen_optimizer.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(gen_optimizer)
            
            scaler.update()

            running_D += loss_D.item()
            running_G_GAN += loss_GAN.item()
            running_L1 += loss_L1.item()

            if i % 100 == 99:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {i+1} | D_Loss: {running_D/100:.4f} | G_GAN: {running_G_GAN/100:.4f} | L1: {running_L1/100:.4f}")
                running_D = running_G_GAN = running_L1 = 0.0

        torch.save(generator.state_dict(), "LearnVoice/generator1.pth")
        torch.save(discriminator.state_dict(), "LearnVoice/discriminator1.pth")