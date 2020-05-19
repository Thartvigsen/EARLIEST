import torch
from torch import nn
from modules import Controller, BaselineNetwork
import numpy as np

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
    ninp : int
        number of features in the input data.
    nclasses : int
        number of classes in the input labels.
    nhid : int
        number of dimensions in the RNN's hidden states.
    rnn_type : str
        which RNN memory cell to use: {LSTM, GRU, RNN}.
        (if defining your own, leave this alone)
    lam : float32
        earliness weight -- emphasis on earliness.
    nlayers : int
        number of layers in the RNN.

    """
    def __init__(self, ninp=1, nclasses=1, nhid=50, rnn_type="LSTM",
                 nlayers=1, lam=0.0):
        super(EARLIEST, self).__init__()

        # --- Hyperparameters ---
        self.ninp = ninp
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.lam = lam
        self.nclasses = nclasses

        # --- Sub-networks ---
        self.Controller = Controller(nhid+1, 1)
        self.BaselineNetwork = BaselineNetwork(nhid+1, 1)
        if rnn_type == "LSTM":
            self.RNN = torch.nn.LSTM(ninp, nhid)
        else:
            self.RNN = torch.nn.GRU(ninp, nhid)
        self.out = torch.nn.Linear(nhid, nclasses)

    def initHidden(self, bsz):
        """Initialize hidden states"""
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.nlayers, bsz, self.nhid),
                    torch.zeros(self.nlayers, bsz, self.nhid))
        else:
            return torch.zeros(self.nlayers, bsz, self.nhid)

    def forward(self, X, epoch=0, test=False):
        """Compute halting points and predictions"""
        if test: # Model chooses for itself during testing
            self.Controller._epsilon = 0.0
        else:
            self.Controller._epsilon = self._epsilon # explore/exploit trade-off
        T, B, V = X.shape
        baselines = [] # Predicted baselines
        actions = [] # Which classes to halt at each step
        log_pi = [] # Log probability of chosen actions
        halt_probs = []
        halt_points = -torch.ones((B, self.nclasses))
        hidden = self.initHidden(X.shape[1])
        predictions = torch.zeros((B, self.nclasses), requires_grad=True)
        all_preds = []

        # --- for each timestep, select a set of actions ---
        for t in range(T):
            # run Base RNN on new data at step t
            RNN_in = X[t].unsqueeze(0)
            output, hidden = self.RNN(RNN_in, hidden)

            # predict logits for all elements in the batch
            logits = self.out(output.squeeze())

            # compute halting probability and sample an action
            time = torch.tensor([t], dtype=torch.float, requires_grad=False).view(1, 1, 1).repeat(1, B, 1)
            c_in = torch.cat((output, time), dim=2).detach()
            a_t, p_t, w_t = self.Controller(c_in)

            # If a_t == 1 and this class hasn't been halted, save its logits
            predictions = torch.where((a_t == 1) & (predictions == 0), logits, predictions)

            # If a_t == 1 and this class hasn't been halted, save the time
            halt_points = torch.where((halt_points == -1) & (a_t == 1), time.squeeze(0), halt_points)

            # compute baseline
            b_t = self.BaselineNetwork(torch.cat((output, time), dim=2).detach())

            actions.append(a_t.squeeze())
            baselines.append(b_t.squeeze())
            log_pi.append(p_t)
            halt_probs.append(w_t)
            if (halt_points == -1).sum() == 0:  # If no negative values, every class has been halted
                break

        # If one element in the batch has not been halting, use its final prediction
        logits = torch.where(predictions == 0.0, logits, predictions).squeeze()
        halt_points = torch.where(halt_points == -1, time, halt_points).squeeze(0)
        self.locations = np.array(halt_points + 1)
        self.baselines = torch.stack(baselines).squeeze(1).transpose(0, 1)
        self.log_pi = torch.stack(log_pi).squeeze(1).squeeze(2).transpose(0, 1)
        self.halt_probs = torch.stack(halt_probs).transpose(0, 1).squeeze(2)
        self.actions = torch.stack(actions).transpose(0, 1)

        # --- Compute mask for where actions are updated ---
        # this lets us batch the algorithm and just set the rewards to 0
        # when the method has already halted one instances but not another.
        self.grad_mask = torch.zeros_like(self.actions)
        for b in range(B):
            self.grad_mask[b, :(1 + halt_points[b, 0]).long()] = 1
        return logits.squeeze(), (1+halt_points).mean()/(T+1)

    def computeLoss(self, logits, y):
        # --- compute reward ---
        _, y_hat = torch.max(torch.softmax(logits, dim=1), dim=1)
        self.r = (2*(y_hat.float().round() == y.float()).float()-1).detach().unsqueeze(1)
        self.R = self.r * self.grad_mask

        # --- rescale reward with baseline ---
        b = self.grad_mask * self.baselines
        self.adjusted_reward = self.R - b.detach()

        # If you want a discount factor, that goes here!
        # It is used in the original implementation.

        # --- compute losses ---
        MSE = torch.nn.MSELoss()
        CE = torch.nn.CrossEntropyLoss()
        self.loss_b = MSE(b, self.R) # Baseline should approximate mean reward
        self.loss_r = (-self.log_pi*self.adjusted_reward).sum()/self.log_pi.shape[1] # RL loss
        self.loss_c = CE(logits, y) # Classification loss
        self.wait_penalty = self.halt_probs.sum(1).mean() # Penalize late predictions
        self.lam = torch.tensor([self.lam], dtype=torch.float, requires_grad=False)
        loss = self.loss_r + self.loss_b + self.loss_c + self.lam*(self.wait_penalty)
        # It can help to add a larger weight to self.loss_c so early training
        # focuses on classification: ... + 10*self.loss_c + ...
        return loss
