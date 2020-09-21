# Import Module
import pickle
import numpy as np

# Import PyTorch
import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path,'rb') as f:
            self.input_=np.array(pickle.load(f))
            self.weekday_=np.array(pickle.load(f))
            self.hour_=np.array(pickle.load(f))
            self.output_=np.array(pickle.load(f))
        self.num_data = len(self.input_)
        

    def __getitem__(self, index):
        src = self.input_[index]
        hour = self.input_[index]
        weekday = self.input_[index]
        src_rev = np.flip(src)
        trg = self.output_[index]
        return src, src_rev, weekday, hour, trg
    
    def __len__(self):
        return self.num_data



class Transpose_tensor:
    def __init__(self, dim=1):
        self.dim = dim

    def transpose_tensor(self, batch):
        (src, src_rev, weekday, hour, trg) = zip(*batch)
        batch_size = len(src)

        src_t = torch.FloatTensor(src)
        src_rev_t = torch.FloatTensor(src_rev)
        src_hour_t = torch.LongTensor(hour)
        src_weekday_t = torch.LongTensor(weekday)
        trg_t = torch.LongTensor(trg)
        #..#
        # src = torch.cat(src).view(-1, batch_size).transpose(0, 1)
        # src_rev = torch.cat(src_rev, dim=self.dim).view(-1, batch_size).transpose(0, 1)
        # trg = torch.cat(trg).view(-1, batch_size).transpose(0, 1)

        return src_t, src_rev_t, src_hour_t, src_weekday_t, trg_t

    def __call__(self, batch):
        return self.transpose_tensor(batch)

def getDataLoader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last):
    return DataLoader(dataset, drop_last=drop_last, batch_size=batch_size, collate_fn=Transpose_tensor(),
                      shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers) 
