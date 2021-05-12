from torchvision import datasets, transforms
from base import BaseDataLoader

import torch
from utils.dist import get_world_size


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # import pdb
        # pdb.set_trace()
        
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        if get_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=shuffle)
        else:
            sampler = None
        super().__init__(self.dataset, batch_size, shuffle, num_workers, sampler=sampler)
