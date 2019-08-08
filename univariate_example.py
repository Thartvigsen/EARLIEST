import numpy as np
import torch
from model import EARLIEST
from dataset import SyntheticTimeSeries
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- methods ---
def exponentialDecay(N):
    tau = 1 
    tmax = 4 
    t = np.linspace(0, tmax, N)
    y = np.exp(-t/tau)
    y = torch.FloatTensor(y)
    return y/10.

# --- hyperparameters ---
HIDDEN_DIMENSION = 10
N_FEATURES = 1
N_LAYERS = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
CELL_TYPE = "LSTM"
N_EPOCHS = 20
SEQ_LENGTH = 10
LAMBDA = 1e-02
exponentials = exponentialDecay(N_EPOCHS)

# --- dataset ---
data = SyntheticTimeSeries(T=SEQ_LENGTH)
N_CLASSES = data.N_CLASSES # Number of classes for classification
train_sampler = SubsetRandomSampler(data.train_ix)
test_sampler = SubsetRandomSampler(data.test_ix)
train_loader = torch.utils.data.DataLoader(dataset=data,
                                           batch_size=BATCH_SIZE, 
                                           sampler=train_sampler,
                                           drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=data,
                                          batch_size=BATCH_SIZE, 
                                          sampler=test_sampler,
                                          drop_last=True)

# --- initialize the model and the optimizer ---
model = EARLIEST(N_FEATURES, N_CLASSES, HIDDEN_DIMENSION, CELL_TYPE, N_LAYERS, LAMBDA=LAMBDA)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# --- training ---
training_loss = []
training_locations = []
training_predictions = []
for epoch in range(N_EPOCHS):
    model._REWARDS = 0
    model._r_sums = np.zeros(SEQ_LENGTH).reshape(1, -1)
    model._r_counts = np.zeros(SEQ_LENGTH).reshape(1, -1)
    model._epsilon = exponentials[epoch]
    loss_sum = 0
    for i, (X, y) in enumerate(train_loader):
        X = torch.transpose(X, 0, 1)
        # --- Forward pass ---
        predictions = model(X)

        # --- Compute gradients and update weights ---
        optimizer.zero_grad()
        loss = model.applyLoss(predictions, y)
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()
        #scheduler.step()

        # --- Collect prediction locations ---
        for j in range(len(y)):
            training_locations.append(model.locations[j])
            training_predictions.append(predictions[j])
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, N_EPOCHS, i+1, len(train_loader), loss.item()))
    training_loss.append(np.round(loss_sum/len(train_loader), 3))
training_locations = torch.stack(training_locations).numpy()

# --- testing ---
testing_loss = []
testing_predictions = []
testing_labels = []
testing_locations = []
loss_sum = 0
for i, (X, y) in enumerate(test_loader):
    X = torch.transpose(X, 0, 1)
    predictions = model(X)
    for j in range(len(y)):
        testing_locations.append(model.locations[j])
        testing_predictions.append(predictions[j])
        testing_labels.append(y[j])
    loss = model.applyLoss(predictions, y)
    loss.backward()
    loss_sum += loss.item()
    testing_loss.append(np.round(loss_sum/len(test_loader), 3))
_, testing_predictions = torch.max(torch.stack(testing_predictions).detach(), 1)
testing_predictions = np.array(testing_predictions)

print("Accuracy: {}".format(accuracy_score(testing_labels, testing_predictions)))
print("Mean proportion used: {}%".format(np.round(np.mean(testing_locations)/SEQ_LENGTH, 3)))
