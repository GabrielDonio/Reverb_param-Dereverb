import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Sampler
import glob
import random
class DataReverb(data.Dataset):
    def __init__(self, files1, files2, transform=None,N=4000):
        self.files1 = files1
        self.files2 = files2
        self.transform = transform
        self.N = N

    def __len__(self):
        return len(self.files1)

    def __getitem__(self, idx):
        data1 = torch.load(self.files1[idx], weights_only=True)
        data2 = torch.load(self.files2[idx], weights_only=True)
        if self.transform:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
        data1 = data1[self.N:]
        data2 = data2[self.N:]
        return data1, data2

def collate_pad(batch):
    data1, data2 = zip(*batch)
    lengths = [x.shape[0] for x in data1]
    min_len = min(lengths)
    data1_crop = [x[:min_len] for x in data1]
    data2_crop = [x[:min_len] for x in data2]
    return torch.stack(data1_crop), torch.stack(data2_crop), torch.tensor(lengths)

class ShuffledBatchSampler(Sampler):
    def __init__(self, batches, shuffle=True):
        self.batches = batches
        self.shuffle = shuffle

    def __iter__(self):
        batches = self.batches.copy()
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches)

if __name__ == "__main__":
    clean_files = sorted(glob.glob("preprocessed/clean/*.pt"))
    reverb_files = sorted(glob.glob("preprocessed/reverb/*.pt"))
    lengths = torch.load("preprocessed/lengths.pt", weights_only=True)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    files1_sorted = [clean_files[i] for i in sorted_indices]
    files2_sorted = [reverb_files[i] for i in sorted_indices]

    dataset = DataReverb(files1_sorted, files2_sorted)

    batch_size = 8

    batches = [
        list(range(i, min(i + batch_size, len(dataset))))
        for i in range(0, len(dataset), batch_size)
    ]

    batch_sampler = ShuffledBatchSampler(batches, shuffle=True)


    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_pad,
    )
    for epoch in range(2):
        print(f"\nEpoch {epoch}")
        for i, (clean_batch, reverb_batch, lengths_batch) in enumerate(loader):
            print(f"Batch {i}: clean shape {clean_batch.shape}, reverb shape {reverb_batch.shape}, longueurs {lengths_batch.tolist()}")
            if i == 5000:
                break