import torch
from SpectrogramDataset import SpectrogramDataset
from SimpleDisc import Pix2pixDiscriminator
from TransformModel import ConformerModel
import torch.utils.data as data
import torch.optim as optim
import os

# Hyperparameters
MIN_LENGTH = 137
BATCH_SIZE = 16
LEARNING_RATE_GEN = 1e-4        
LEARNING_RATE_DISC = 1e-4       
GAN_LAMBDA = 5               
L1_LAMBDA = 50
DISC_SMOOTHING = 0.9
FAKE_DISC_SMOOTHING = 0.05
EPOCHS = 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    CleanDir = "spec/clean_spectrograms/"
    NoisyDir = "spec/noisy_spectrograms/"
    dataset = SpectrogramDataset(CleanDir, NoisyDir, N=MIN_LENGTH,Nb_files=16000)
    torch.backends.cudnn.benchmark = True
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=8, pin_memory=True, prefetch_factor=2)

    # Models
    generator = ConformerModel(freqdim=256, heads=8, ffn_dim=192, depth=3, conv_kernel_size=41, dropout=0.1).to(device)
    discriminator = Pix2pixDiscriminator(conv_kernel_size=5,ffn_dim=64).to(device)
    
    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_GEN)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_DISC)
    
    if os.path.exists("LearnVoice/generator.pth"):
        generator.load_state_dict(torch.load("LearnVoice/generator.pth", map_location=device))
    if os.path.exists("LearnVoice/discriminator.pth"):
        discriminator.load_state_dict(torch.load("LearnVoice/discriminator.pth", map_location=device))
    
    L1_loss = torch.nn.L1Loss()
    criterion_GAN = torch.nn.MSELoss() 
    
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()
    
    generator.train()
    discriminator.train()
    running_D = 0.0
    running_G = 0.0
    running_L1 = 0.0
    print_count = 0
    
    for epoch in range(EPOCHS):
        for i, (noisy_spec, clean_spec) in enumerate(dataloader):
            noisy_spec = noisy_spec.to(device)
            clean_spec = clean_spec.to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            if i % 8 == 0:
                with torch.amp.autocast(device_type=device.type):
                    # Génération
                    mask = generator(noisy_spec)         
                    gen_spec = noisy_spec*mask     
                    
                    D_real = discriminator(clean_spec, noisy_spec)   
                    D_fake = discriminator(gen_spec.detach(), noisy_spec)  
                        
                        # Smooth real labels si nécessaire
                    real_labels = torch.ones_like(D_real) * DISC_SMOOTHING
                    fake_labels = torch.zeros_like(D_fake) + FAKE_DISC_SMOOTHING
                        
                    loss_D_real = criterion_GAN(D_real, real_labels)
                    loss_D_fake = criterion_GAN(D_fake, fake_labels)
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)
                    
                disc_optimizer.zero_grad()
                d_scaler.scale(loss_D).backward()
                d_scaler.step(disc_optimizer)
                d_scaler.update()
                running_D += loss_D.item()
            
            # -----------------
            #  Train Generator
            # -----------------
            with torch.amp.autocast(device_type=device.type):
                mask = generator(noisy_spec)
                gen_spec = noisy_spec*mask     
                    
                D_fake = discriminator(gen_spec, noisy_spec)
                
                # Loss adversariale (on veut que D dise 1)
                real_labels_for_G = torch.ones_like(D_fake)* DISC_SMOOTHING
                loss_GAN = criterion_GAN(D_fake, real_labels_for_G)
                
                # Loss reconstruction L1
                loss_L1 = L1_loss(gen_spec, clean_spec)
                # Perte totale
                loss_G = GAN_LAMBDA * loss_GAN + L1_LAMBDA * loss_L1 
            
            gen_optimizer.zero_grad()
            g_scaler.scale(loss_G).backward()
            g_scaler.step(gen_optimizer)
            g_scaler.update()
            
            running_G += loss_GAN.item()
            running_L1 += loss_L1.item()
            print_count += 1
            if print_count == 150:
                print(f"Avg over 150 batches — Loss D: {running_D/(150/8):.6f}, Loss G: {running_G/150:.6f}, L1: {running_L1/150:.6f}")
                running_D = running_G = running_L1 = 0.0
                print_count = 0
 
    # Save the models
    torch.save(generator.state_dict(), "LearnVoice/generator.pth")    
    torch.save(discriminator.state_dict(), "LearnVoice/discriminator.pth")
