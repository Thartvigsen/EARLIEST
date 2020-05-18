import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticTimeSeries(Dataset):
    def __init__(self, T=10):
        self.N = 500
        self.T = T
        self.N_FEATURES = 1
        self.data, self.labels, self.signal_locs = self.generate()
        self.train_ix, self.test_ix = self.getSplitIndices()
        self.N_CLASSES = len(np.unique(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

    def getSplitIndices(self):
        split_props = [0.8, 0.2]
        indices = np.arange(self.N)
        split_points = [int(self.N*i) for i in split_props]
        train_ix = np.random.choice(indices,
                                    split_points[0],
                                    replace=False)
        test_ix = np.random.choice((list(set(indices)-set(train_ix))),
                                   split_points[1],
                                   replace=False)
        return train_ix, test_ix

    def generate(self):
        self.signal_locs = np.random.randint(self.T, size=int(self.N))
        X = np.zeros((self.N, self.T, 1))
        y = np.zeros((self.N))

        for i in range(int(self.N)):
            if i < (int(self.N/2.)):
                X[i, self.signal_locs[i], 0] = 1
                y[i] = 1
            else:
                X[i, self.signal_locs[i], 0] = 0

        self.signal_locs[int(self.N/2):] = -1 
        data = torch.tensor(np.asarray(X).astype(np.float32),
                            dtype=torch.float)
        labels = torch.tensor(np.array(y).astype(np.int32), dtype=torch.long)
        signal_locs = torch.tensor(np.asarray(self.signal_locs),
                                   dtype=torch.float)
        return data, labels, signal_locs
