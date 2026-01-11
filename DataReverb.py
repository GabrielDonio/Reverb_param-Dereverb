import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import glob


clean_files = sorted(glob.glob("preprocessed/clean/*.pt"))
reverb_files = sorted(glob.glob("preprocessed/reverb/*.pt"))
lengths = torch.load("preprocessed/lengths.pt", weights_only=True)
count = sum(1 for l in lengths if int(l) <= 60000)
print(count/len(lengths))
