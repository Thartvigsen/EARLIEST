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

    def forward(self, x):
        b = self.fc(x.detach())
        return b

class Controller(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, ninp, nout):
        super(Controller, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)  # Optimized w.r.t. reward

    def forward(self, x):
        probs = torch.sigmoid(self.fc(x))
        probs = (1-self._epsilon)*probs + self._epsilon*torch.FloatTensor([0.05])  # Explore/exploit
        m = Bernoulli(probs=probs)
        action = m.sample() # sample an action
        log_pi = m.log_prob(action) # compute log probability of sampled action
        return action.squeeze(0), log_pi.squeeze(0), -torch.log(probs).squeeze(0)
