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
LEARNING_RATE = 1e-2
BATCH_SIZE = 32
CELL_TYPE = "LSTM"
N_EPOCHS = 100
SEQ_LENGTH = 10
LAMBDA = 0.0
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
model = EARLIEST(ninp=N_FEATURES, nclasses=N_CLASSES, nhid=HIDDEN_DIMENSION,
                 rnn_type=CELL_TYPE, nlayers=N_LAYERS, lam=LAMBDA)
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

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, N_EPOCHS, i+1, len(train_loader), loss.item()))
    training_loss.append(np.round(loss_sum/len(train_loader), 3))
    scheduler.step()
training_locations = torch.stack(training_locations).numpy()

import matplotlib.pyplot as plt
plt.plot(training_loss)
plt.show()

# --- testing ---
testing_loss = []
testing_predictions = []
testing_labels = []
testing_locations = []
loss_sum = 0
for i, (X, y) in enumerate(test_loader):
    X = torch.transpose(X, 0, 1)
    logits, halting_points = model(X, test=True)
    _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)

    testing_locations.append(halting_points)
    testing_predictions.append(predictions)
    testing_labels.append(y)

    loss = model.computeLoss(logits, y)
    loss_sum += loss.item()
    testing_loss.append(np.round(loss_sum/len(test_loader), 3))

testing_predictions = torch.stack(testing_predictions).numpy().reshape(-1, 1)
testing_labels = torch.stack(testing_labels).numpy().reshape(-1, 1)
testing_locations = torch.stack(testing_locations).numpy().reshape(-1, 1)
#_, testing_predictions = torch.max(torch.stack(testing_predictions).detach(), 1)
#testing_predictions = np.array(testing_predictions)

print("Accuracy: {}".format(np.round(accuracy_score(testing_labels, testing_predictions), 3)))
print("Mean proportion used: {}%".format(np.round(100.*np.mean(testing_locations), 3)))
