# Num-Workers-Search
*num_workers* search algorithm for fast PyTorch DataLoader

# Background

To find **optimal** *num_workers* for PyTorch DataLoader is the key towards fast training.

I measured the total time for loading all the training data of CIFAR10 Dataset with various *num_workers* size on my PC and Google's Colab.

## on my PC

![image](https://user-images.githubusercontent.com/35001605/127024522-42a5ae9a-e93f-423b-9cff-8ded69809547.png)

### Spec
- CPU: i5-10400
- GPU: RTX 3070
- RAM:32GB
- OS: Windows 10
- SSD: 1TB

## on Colab

![image](https://user-images.githubusercontent.com/35001605/127024889-2bebfebb-bc35-46d2-ac14-70288790e461.png)

As you see, to find optimal *num_workers* is very important for fast training. But it is hard to pick optimal *num_workers* by some formular because it varies with pc spec, dataset size, and batch size.

# My Solution

I thought that we can find optimal *num_workers* through full search algorithm. I added some early stopping condition for speedup of this process. It is inspired from the learning rate scheduler [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau).

```python
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
```

This method needs to spend time to search optimal *num_workers* but it can significantly save the entire training time!

# In-script workflow

```python
import torch
import nws

batch_size = ...
dataset = ...

optimal_num_workers = nws.search(dataset=dataset,
                                 batch_size=batch_size,
                                 ...)

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size, 
                                     ...,
                                     num_workers=optimal_num_workers, 
                                     ...)
```
