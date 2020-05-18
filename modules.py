from torch import nn
import torch
from torch.distributions import Bernoulli

class BaselineNetwork(nn.Module):
    """
    A network which predicts the average reward observed
    during a markov decision-making process.
    Weights are updated w.r.t. the mean squared error between
    its prediction and the observed reward.
    """

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        #         b_t = torch.relu(z_t)
        return b_t

class Controller(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, INPUT_DIM, N_CLASSES):
        super(Controller, self).__init__()

        # --- Hyperparameters ---
        self.ADDITIVE = False  # Only increase halting probs? Or increase/decrease?

        # --- Mappings ---
        self.fc = nn.Linear(INPUT_DIM, N_CLASSES)  # Optimized w.r.t. reward
        self.boost = nn.Linear(N_CLASSES, N_CLASSES)

        # --- Nonlinearities ---
        self.tanh = nn.Tanh()

    def forward(self, x, eps=0.):
        probs = self.fc(x)

        # --- tie labels together according to predicted likelihoods ---
        #         if self.ADDITIVE:
        #             probs = torch.relu(self.boost(probs)) + probs # Increase halting-probabilities depending on halting-probabilities
        #         else:
        #             probs = torch.tanh(self.boost(probs)) + probs # Change halting-probabilities depending on other likelihoods
        #         probs = torch.sigmoid(self.fc(hidden_locked)) # Compute halting-probability

        probs = torch.sigmoid(probs)  # Rescale back to probabilistic space
        probs = (1-self._epsilon)*probs + self._epsilon*torch.FloatTensor([0.05])  # Explore/exploit (can't be 0)
        m = Bernoulli(probs=probs)
        action = m.sample() # sample an action
        log_pi = m.log_prob(action) # compute log probability of sampled action
        return action.squeeze(0), log_pi.squeeze(0), -torch.log(probs).squeeze(0)
