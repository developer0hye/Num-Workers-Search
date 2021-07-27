import torch
import numpy as np
import os
import time

def search(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False, maximum_patience=3):
    
    max_num_workers = os.cpu_count() #refer to https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py    

    optimal_num_workers = 0
    min_total_time = np.finfo(np.float).max
    patience = 0
    
    for num_workers in np.arange(max_num_workers + 1):
        loader = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle,
                                            sampler=sampler,
                                            batch_sampler=batch_sampler,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory,
                                            drop_last=drop_last,
                                            timeout=timeout,
                                            worker_init_fn=worker_init_fn,
                                            prefetch_factor=prefetch_factor,
                                            persistent_workers=persistent_workers)

        t1 = time.time()
        for _ in loader: pass
        t2 = time.time()
        
        total_time = t2 - t1
        
        if min_total_time > total_time:
            optimal_num_workers = num_workers
            min_total_time = total_time
            patience = 0
        else:
            patience += 1
            if patience > maximum_patience: # early stopping condition
                break

    return optimal_num_workers
