import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from modules import BaseRNN, Controller
from modules import BaselineNetwork, Discriminator

class EARLIEST(nn.Module):
    """Code for the paper titled: Adaptive-Halting Policy Network for Early Classification
    Paper link: https://dl.acm.org/citation.cfm?id=3330974

    Method at a glance: An RNN is trained to model time series
    with respect to a classification task. A controller network
    decides at each timestep whether or not to generate the
    classification. Once a classification is made, the RNN
    stops processing the time series.

    Parameters
    ----------
    N_FEATURES : int
        number of features in the input data.
    N_CLASSES : int
        number of classes in the input labels.
    HIDDEN_DIM : int
        number of dimensions in the RNN's hidden states.
    CELL_TYPE : str
        which RNN memory cell to use: {LSTM, GRU, RNN}.
        (if defining your own, leave this alone)
    DF : float32
        discount factor for optimizing the Controller.
    LAMBDA : float32
        earliness weight -- emphasis on earliness.
    N_LAYERS : int
        number of layers in the RNN.

    """
    def __init__(self, N_FEATURES=1, N_CLASSES=2, HIDDEN_DIM=50, CELL_TYPE="LSTM",
                 DF=1., LAMBDA=1.0, N_LAYERS=1):
        super(EARLIEST, self).__init__()

        # --- Hyperparameters ---
        self.CELL_TYPE = CELL_TYPE
        self.HIDDEN_DIM = HIDDEN_DIM
        self.DF = DF
        self.LAMBDA = LAMBDA
        self.N_LAYERS = N_LAYERS
        self._epsilon = 1.0
        self._rewards = 0

        # --- Sub-networks ---
        self.BaseRNN = BaseRNN(N_FEATURES,
                               HIDDEN_DIM,
                               CELL_TYPE)
        self.Controller = Controller(HIDDEN_DIM+1, 1) # Add +1 for timestep input
        self.BaselineNetwork = BaselineNetwork(HIDDEN_DIM, 1)
        self.Discriminator = Discriminator(HIDDEN_DIM, N_CLASSES)

    def initHidden(self, batch_size):
        """Initialize hidden states"""
        if self.CELL_TYPE == "LSTM":
            h = (torch.zeros(self.N_LAYERS,
                             batch_size,
                             self.HIDDEN_DIM),
                 torch.zeros(self.N_LAYERS,
                             batch_size,
                             self.HIDDEN_DIM))
        else:
            h = torch.zeros(self.N_LAYERS,
                            batch_size,
                            self.HIDDEN_DIM)
        return h

    def forward(self, X):
        baselines = []
        log_pi = []
        halt_probs = []
        attention = []
        hidden_states = []
        hidden = self.initHidden(X.shape[1]) # Initialize hidden states - input is batch size
        for t in range(len(X)):
            x_t = X[t].unsqueeze(0) # add time dim back in
            S_t, hidden = self.BaseRNN(x_t, hidden) # Run sequence model
            S_t = S_t.squeeze(0) # remove time dim
            t = torch.tensor([t], dtype=torch.float).view(1, 1) # collect timestep
            hidden_states.append(S_t)
            S_t_with_t = torch.cat((S_t, t), dim=1) # Add timestep as input to controller
            a_t, p_t, w_t = self.Controller(S_t_with_t, self._epsilon) # Compute halting-probability and sample an action
            b_t = self.BaselineNetwork(S_t) # Compute the baseline
            baselines.append(b_t)
            log_pi.append(p_t)
            halt_probs.append(w_t)
            if a_t == 1:
                break

        y_hat = self.Discriminator(S_t) # Classify the time series
        self.baselines = torch.stack(baselines).transpose(1, 0)
        self.baselines = self.baselines.view(1, -1)
        self.log_pi = torch.stack(log_pi).transpose(1, 0).squeeze(2)
        self.halt_probs = torch.stack(halt_probs).transpose(1, 0)
        self.halting_point = t+1 # Adjust timestep indexing just for plotting
        self.locations = self.halting_point
        return y_hat
               
    def applyLoss(self, y_hat, labels):
        # --- compute reward ---
        _, predicted = torch.max(y_hat, 1)
        r = (predicted.float().detach() == labels.float()).float()
        r = r*2 - 1
        R = r.unsqueeze(1).repeat(1, int(self.halting_point.squeeze()))

        # --- discount factor ---
        discount = [self.df**i for i in range(self.halting_point)]
        discount = np.array(discount).reshape(1, -1)
        discount = np.flip(discount, 1)
        discount = torch.from_numpy(discount.copy()).float().view(1, -1)
        R = R * discount
        self._rewards += torch.sum(R) # Collect the sum of rewards for plotting

        # --- subtract baseline from reward ---
        adjusted_reward = R - self.baselines.detach()

        # --- compute losses ---
        self.loss_b = F.mse_loss(self.baselines, R) # Baseline should approximate mean reward
        self.loss_c = F.cross_entropy(logits, labels) # Make accurate predictions
        self.loss_r = torch.sum(-self.log_pi*adjusted_reward, dim=1) # Controller should lead to correct predictions from the discriminator
        self.lam = torch.tensor([self.lam], requires_grad=False)
        self.time_penalty = torch.sum(self.halt_probs, dim=1) # Penalize late predictions

        # --- collect all loss terms ---
        loss = (self.loss_r \
                + self.loss_c \
                + self.loss_b \
                + self.lam*self.time_penalty)
        return loss
