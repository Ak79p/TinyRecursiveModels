import json
import torch
from torch.utils.data import Dataset, DataLoader

class SudokuDataset(Dataset):
    def __init__(self, path):
        self.items = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        obj = self.items[idx]
        inp = torch.tensor(obj["input"], dtype=torch.long)
        tgt = torch.tensor(obj["target"], dtype=torch.long)
        return inp, tgt, obj["id"]

def get_sudoku_loader(path, batch_size=1):
    ds = SudokuDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

if __name__ == "__main__":
    loader = get_sudoku_loader("data/sudoku_small.jsonl", batch_size=1)
    for i, (inp, tgt, idx) in enumerate(loader):
        print("Batch", i, "| id =", idx.item())
        print("Input shape:", inp.shape)
        print("Target shape:", tgt.shape)
        if i == 2:
            break