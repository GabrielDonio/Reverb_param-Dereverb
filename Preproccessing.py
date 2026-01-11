import os
import glob
import torch
import torchaudio

# =========================
# CONFIG
# =========================
clean_path = "clean_trainset_28spk_wav"
reverb_path = "reverb_trainset_28spk_wav"
out_path = "preprocessed"

target_sr = 16000
crop_begin = 4000
crop_end = 500

os.makedirs(out_path + "/clean", exist_ok=True)
os.makedirs(out_path + "/reverb", exist_ok=True)

clean_files = sorted(glob.glob(clean_path + "/*.wav"))
reverb_files = sorted(glob.glob(reverb_path + "/*.wav"))

assert len(clean_files) == len(reverb_files)

lengths = []

print(f"Preprocessing {len(clean_files)} files...")

for i, (cf, rf) in enumerate(zip(clean_files, reverb_files)):
    clean, sr1 = torchaudio.load(cf)
    reverb, sr2 = torchaudio.load(rf)

    if sr1 != target_sr:
        clean = torchaudio.functional.resample(clean, sr1, target_sr)
    if sr2 != target_sr:
        reverb = torchaudio.functional.resample(reverb, sr2, target_sr)

    # mono
    clean = clean[0]
    reverb = reverb[0]

    # crop
    T = min(clean.shape[0], reverb.shape[0])
    clean = clean[:T]
    reverb = reverb[:T]

    clean = clean[crop_begin:T - crop_end]
    reverb = reverb[crop_begin:T - crop_end]

    assert clean.shape[0] > 0

    # save
    name = f"{i:06d}.pt"
    torch.save(clean, f"{out_path}/clean/{name}")
    torch.save(reverb, f"{out_path}/reverb/{name}")

    lengths.append(clean.shape[0])

    if i % 500 == 0:
        print(f"  processed {i}/{len(clean_files)}")

torch.save(lengths, f"{out_path}/lengths.pt")

print("✅ Preprocessing terminé")
