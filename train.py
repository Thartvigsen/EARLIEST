import numpy as np
import argparse
import torch
from model import EARLIEST
from dataset import SyntheticTimeSeries
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Dataset hyperparameters
parser.add_argument("--dataset", type=str, help="Dataset to load. Available: Synthetic")
parser.add_argument("--ntimesteps", type=int, default=10, help="Synthetic dataset can control the number of timesteps")
parser.add_argument("--nseries", type=int, default=500, help="Synthetic dataset can control the number of time series")

# Model hyperparameters
parser.add_argument("--nhid", type=int, default=50, help="Number of dimensions of the hidden state of EARLIEST")
parser.add_argument("--nlayers", type=int, default=1, help="Number of layers for EARLIEST's RNN.")
parser.add_argument("--rnn_cell", type=str, default="LSTM", help="Type of RNN to use in EARLIEST. Available: GRU, LSTM")
parser.add_argument("--lambda", type=float, default=0.0, help="Penalty of waiting. This controls the emphasis on earliness: Larger values lead to earlier predictions.")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, help="Batch size.")
parser.add_argument("--nepochs", type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", type=float, default="0.001", help="Learning rate.")
parser.add_argument("--model_save_path", type=str, default="./saved_models/", help="Where to save the model once it is trained.")

args = parser.parse_args()

if name == "__main__":
    model_save_path = args.model_save_path

    if args.dataset == "Synthetic":
        data = SyntheticTimeSeries(args)
    train_ix, val_ix, test_ix = utils.splitTrainingData(data.nseries)

    train_sampler = SubsetRandomSampler(train_ix)
    validation_sampler = SubsetRandomSampler(validation_ix)
    test_sampler = SubsetRandomSampler(test_ix)

    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset=data,
                                                    batch_size=args.batch_size,
                                                    sampler=validation_sampler,
                                                    drop_last=True)

    model = EARLIEST(ninp=data.N_FEATURES, nclasses=data.N_CLASSES, args) #nhid=HIDDEN_DIMENSION, rnn_type=CELL_TYPE, nlayers=N_LAYERS, lam=LAMBDA)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # --- training ---
    training_loss = []
    training_locations = []
    training_predictions = []
    for epoch in range(args.nepochs):
        model._REWARDS = 0
        model._r_sums = np.zeros(SEQ_LENGTH).reshape(1, -1)
        model._r_counts = np.zeros(SEQ_LENGTH).reshape(1, -1)
        model._epsilon = exponentials[epoch]
        loss_sum = 0
        for i, (X, y) in enumerate(train_loader):
            X = torch.transpose(X, 0, 1)
            # --- Forward pass ---
            logits, halting_points = model(X, epoch)
            _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

            training_locations.append(halting_points)
            training_predictions.append(predictions)

            # --- Compute gradients and update weights ---
            optimizer.zero_grad()
            loss = model.computeLoss(logits, y)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

        training_loss.append(np.round(loss_sum/len(train_loader), 3))
        scheduler.step()
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(i+1, args.nepochs, i+1, len(train_loader), loss.item()))

    # --- Run model on validation data ---
    for i, (X, y) in enumerate(validation_loader):
        X = torch.transpose(X, 0, 1)
        # --- Forward pass ---
        logits, halting_points = model(X, test=True)
        _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

        validation_locations.append(halting_points)
        validation_predictions.append(predictions)

    validation_predictions = torch.stack(validation_predictions).numpy().reshape(-1, 1)
    validation_labels = torch.stack(validation_labels).numpy().reshape(-1, 1)
    validation_locations = torch.stack(validation_locations).numpy().reshape(-1, 1)

    print("Validation Accuracy: {}".format(np.round(accuracy_score(validation_labels, validation_predictions), 3)))
    print("Mean proportion used: {}%".format(np.round(100.*np.mean(validation_locations), 3)))

    # --- save model ---
    torch.save(model.state_dict(), model_save_path+"model.pt")
