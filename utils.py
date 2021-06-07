import numpy as np
import os
import torch

def exponentialDecay(N):
    tau = 1
    tmax = 4
    t = np.linspace(0, tmax, N)
    y = np.exp(-t/tau)
    y = torch.FloatTensor(y)
    return y/10.

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def splitTrainingData(num_timesteps, split_props=[0.8, 0.1, 0.1]):
    """
    Split the dataset into training/validation/testing

    Parameters
    ----------
    split_props : list
        Each float value in the list indicates the proportion of the data to serve
        as training, validation, and testing data. The values must sum to 1.
    """
    indices = np.arange(num_timesteps)
    split_points = [int(num_timesteps*i) for i in split_props]
    train_ix = np.random.choice(indices,
                                split_points[0],
                                replace=False)
    val_ix = np.random.choice(list(set(indices)-set(train_ix)),
                              split_points[1],
                              replace=False)
    test_ix = np.random.choice((list(set(indices)-set(train_ix)-set(val_ix))),
                               split_points[2],
                               replace=False)
    return train_ix, val_ix, test_ix
