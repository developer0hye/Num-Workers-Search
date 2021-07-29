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

# Solution

Trial and Error

It can spend a long time to search optimal num_workers but it will save the entire training time.

# In-script workflow

```python
import torch
import nws

batch_size = ...
dataset = ...

num_workers = nws.search(dataset=dataset,
                                 batch_size=batch_size,
                                 ...)

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size, 
                                     ...,
                                     num_workers=num_workers, 
                                     ...)
```
