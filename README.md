# Num-Workers-Search
num_worker search algorithm for fast PyTorch DataLoader

# Background

To find **optimal** num_worker for PyTorch DataLoader is the key towards fast training.

I simply did an experiment using CIFAR10 Dataset to show the importance of num_worker. 

I measured the total time for loading all the training data of CIFAR10 Dataset with various num_worker size on my PC and Google's Colab.

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

As you see, to find optimal num_worker is very important for fast training. But it is hard to pick optimal num_worker by some formular because it varies with pc spec and batch size.

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
