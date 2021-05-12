import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, collate_fn=default_collate, sampler=None):
        if not sampler is None:
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': False,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }
            super().__init__(sampler=sampler, **self.init_kwargs)
        else:
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': shuffle,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }
            super().__init__(**self.init_kwargs)
