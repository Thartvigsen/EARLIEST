import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticTimeSeries(Dataset):
    def __init__(self, args):
        self.nseries = args.nseries
        self.ntimesteps = args.ntimesteps
        self.data, self.labels, self.signal_locs = self.generateDataset()
        #self.train_ix, self.val_ix, self.test_ix = self.getSplitIndices()
        self.N_FEATURES = 1
        self.N_CLASSES = len(np.unique(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

    def generateDataset(self):
        self.signal_locs = np.random.randint(self.ntimesteps, size=int(self.nseries))
        X = np.zeros((self.nseries, self.ntimesteps, 1))
        y = np.zeros((self.nseries))

        for i in range(int(self.nseries)):
            if i < (int(self.nseries/2.)):
                X[i, self.signal_locs[i], 0] = 1
                y[i] = 1
            else:
                X[i, self.signal_locs[i], 0] = 0

        self.signal_locs[int(self.nseries/2):] = -1 
        data = torch.tensor(np.asarray(X).astype(np.float32),
                            dtype=torch.float)
        labels = torch.tensor(np.array(y).astype(np.int32), dtype=torch.long)
        signal_locs = torch.tensor(np.asarray(self.signal_locs),
                                   dtype=torch.float)
        return data, labels, signal_locs
