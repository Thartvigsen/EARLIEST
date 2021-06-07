import numpy as np
import argparse
import torch
from model import EARLIEST
from dataset import SyntheticTimeSeries
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score
import utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Dataset hyperparameters
parser.add_argument("--dataset", type=str, help="Dataset to load. Available: Synthetic")
parser.add_argument("--ntimesteps", type=int, default=10, help="Synthetic dataset can control the number of timesteps")
parser.add_argument("--nseries", type=int, default=500, help="Synthetic dataset can control the number of time series")

# Model hyperparameters
parser.add_argument("--nhid", type=int, default=50, help="Number of dimensions of the hidden state of EARLIEST")
parser.add_argument("--nlayers", type=int, default=1, help="Number of layers for EARLIEST's RNN.")
parser.add_argument("--rnn_cell", type=str, default="LSTM", help="Type of RNN to use in EARLIEST. Available: GRU, LSTM")
parser.add_argument("--lam", type=float, default=0.0, help="Penalty of waiting. This controls the emphasis on earliness: Larger values lead to earlier predictions.")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
parser.add_argument("--model_save_path", type=str, default="./saved_models/", help="Where to save the model once it is trained.")
parser.add_argument("--random_seed", type=int, default="42", help="Set the random seed.")

args = parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model_save_path = args.model_save_path

    if args.dataset == "synthetic":
        data = SyntheticTimeSeries(args)
    _, _, test_ix = utils.splitTrainingData(data.nseries)

    test_sampler = SubsetRandomSampler(test_ix)

    test_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=args.batch_size,
                                              sampler=test_sampler,
                                              drop_last=True)

    model = EARLIEST(ninp=data.N_FEATURES, nclasses=data.N_CLASSES, args=args) #nhid=HIDDEN_DIMENSION, rnn_type=CELL_TYPE, nlayers=N_LAYERS, lam=LAMBDA)
    model.load_state_dict(torch.load(model_save_path+"model.pt"), strict=False)

    # --- testing ---
    testing_predictions = []
    testing_labels = []
    testing_locations = []
    for i, (X, y) in enumerate(test_loader):
        X = torch.transpose(X, 0, 1)
        logits, halting_points = model(X, test=True)
        _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

        testing_locations.append(halting_points)
        testing_predictions.append(predictions)
        testing_labels.append(y)

    testing_predictions = torch.stack(testing_predictions).numpy().reshape(-1, 1)
    testing_labels = torch.stack(testing_labels).numpy().reshape(-1, 1)
    testing_locations = torch.stack(testing_locations).numpy().reshape(-1, 1)

    print("Testing Accuracy: {}".format(np.round(accuracy_score(testing_labels, testing_predictions), 3)))
    print("Mean proportion used: {}%".format(np.round(100.*np.mean(testing_locations), 3)))
