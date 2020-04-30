import torch
from torch.distributions import Bernoulli
import torch.nn as nn
import numpy as np

# --- EARLIEST sub-networks ---
class BaseRNN(nn.Module):
    """
    A one-step Recurrent Neural Network (RNN). Given new
    inputs and the previous hidden state, compute a new
    hidden state.

    This module will be used in a loop over all timesteps of a
    time series.
    """
    def __init__(self,
                 N_FEATURES,
                 HIDDEN_DIM,
                 CELL_TYPE="LSTM",
                 N_LAYERS=1):
        super(BaseRNN, self).__init__()

        # --- Mappings ---
        if CELL_TYPE in ["RNN", "LSTM", "GRU"]:
            self.rnn = getattr(nn, CELL_TYPE)(N_FEATURES,
                                              HIDDEN_DIM,
                                              N_LAYERS)
        else:
            try: 
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[CELL_TYPE]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was
                                 supplied, options are ['LSTM', 'GRU',
                                 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(N_FEATURES,
                              HIDDEN_DIM,
                              N_LAYERS,
                              nonlinearity=nonlinearity)

        # --- Non-linearities ---
        self.tanh = nn.Tanh()

    def forward(self, x_t, h_t_prev):
        """
        Parameters
        ----------
        x_t : torch.float
            Features input at timestep t.

        h_t_prev : torch.float
            Hidden state from the previous timestep, t-1.

        Returns
        -------
        output : torch.float
            new output vector for use during prediction.
        h_t : torch.float
            new hidden state.
        """
        output, h_t = self.rnn(x_t, h_t_prev)
        return output, h_t

class Controller(nn.Module):
    """
    A network that chooses whether or not enough information
    has been seen to predict a label of a time series.
    """
    def __init__(self, input_size, output_size):
        super(Controller, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.log_sig = nn.LogSigmoid()

    def forward(self, h_t, eps=0.):
        """
        Parameters
        ----------
        h_t : torch.float
            hidden state of RNN at timestep t.
        eps : float
            epsilon -- controls the explore/exploit trade-off during training.
            this value is decreased as training progresses.

        Returns
        -------
        halt : torch.long
            binary halting decision (If 0, wait. If 1: halt).
        log_pi : torch.float
            log probability of the selected action.
        -torch.log(probs) : torch.float
            negative log probability of the halting probability.
        """
        probs = torch.sigmoid(self.fc(h_t.detach())) # Compute halting-probability
        probs = (1-eps) * probs + eps * torch.FloatTensor([0.5]) # Add randomness according to eps
        m = Bernoulli(probs=probs) # Define bernoulli distribution parameterized with predicted probability
        halt = m.sample() # Sample action
        log_pi = m.log_prob(halt) # Compute log probability for optimization
        return halt, log_pi, -torch.log(probs)

class Discriminator(nn.Module):
    """
    A network that classifies a time series given the hidden
    state of a recurrent neural network.

    In principle, this discriminator can take many forms.
    Here, we use a one layer fully connected network to
    maintain simplicity where possible.
    """
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_t):
        """
        Parameters
        ----------
        h_t : torch.tensor
            hidden state of RNN at timestep t.

        Returns
        -------
        y_hat : torch.float
            prediction made by the discriminator. y_hat is in [0, 1].
        """
        y_hat = self.softmax(self.fc(h_t))
        return y_hat

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
        """
        Parameters
        ----------
        h_t : torch.tensor
            hidden state of RNN at timestep t.

        Returns
        -------
        b_t : torch.float
            predicted baseline at timestep t.
        """
        z_t = self.fc(h_t.detach())
        b_t = torch.relu(z_t)
        return b_t 
