import numpy as np
import torch
from torch.utils.data import Dataset


class SymbolicDataset(Dataset):
    def __init__(self,num_samples, noise_level):
        # samples from a uniform distribution
        self.x=np.random.uniform(0,10,(num_samples,1)).astype(np.float32)
        # y = sin⁡(2πx) + cos⁡(3πx)
        self.y=(np.sin(2*np.pi*self.x) + np.cos(3*np.pi*self.x)).astype(np.float32)
        # some noise
        self.y+=noise_level+np.random.randn(*self.y.shape).astype(np.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        x = torch.from_numpy(self.x[idx])
        y= torch.from_numpy(self.y[idx])

        return x,y