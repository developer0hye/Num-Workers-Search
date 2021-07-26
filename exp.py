import torchvision
import torchvision.transforms as transforms
import torch

import numpy as np

from matplotlib import pyplot as plt

import os
import time


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    max_num_workers = os.cpu_count() #refer to https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py
    
    plt.xlabel('num_workers') 
    plt.ylabel('Total Time(Sec)')
    
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_workers_list = np.arange(max_num_workers + 1)
    for batch_size in batch_size_list:
        total_time_per_num_workers = []
        for num_workers in num_workers_list:
            loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=num_workers)
            
            t1 = time.time()
            for _ in loader: pass
            t2 =time.time()
            
            total_time = t2 - t1
            total_time_per_num_workers.append(total_time)
            print(f"batch_size{batch_size}, num_workers{num_workers}, total_time(sec): ", total_time)
        plt.plot(num_workers_list, total_time_per_num_workers)
    plt.legend([f"batch size {batch_size}" for batch_size in batch_size_list])
    plt.show()
    