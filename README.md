# Num-Workers-Search
num_worker search algorithm for fast PyTorch DataLoader

# Background

To find **optimal** num_worker for PyTorch DataLoader is the key towards fast training.

I simply did an experiment using CIFAR10 Dataset to show the importance of num_worker. 

I measured the total time for loading all the training data of CIFAR10 Dataset with various num_worker size on my PC and Google's Colab .






## My PC Spec
- CPU: i5-10400
- GPU: RTX 3070
- RAM:32GB
- OS: Windows 10
- SSD: 1TB

As you see, to find optimal num_worker is very important for fast training and optimal num_worker varies with pc spec.

There are some guidelines to find optimal num_worker based on user's pc spec(prior knowledge). But users will not like the method to need prior knowledge.

We can find optimal num_worker using full search algorithm.

# My Solution
```python
def search(dataset):
  num_workers_list = [0, 1, 2, 4, 8, 16, 32, 64]
  

```

This method needs to spend time to search optimal num_worker but it can significantly save the entire training time!

# In-script workflow

```python
from nws import search


```
