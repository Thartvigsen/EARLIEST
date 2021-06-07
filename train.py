import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Dataset parameters
parser.add_argument('--dataset', type=str, help='Dataset to load. Available: Synthetic')
parser.add_argument('--num_timesteps', type=int, help='Synthetic dataset can control the number of timesteps')

# Model parameters
parser.add_argument('--hidden_dim', type=int, help='Number of dimensions of the hidden state of EARLIEST')
parser.add_argument('--rnn_cell', type=int, help='Type of RNN to use in EARLIEST. Available: GRU, LSTM')
parser.add_argument('--lambda', type=int, help='Penalty of waiting. This controls the emphasis on earliness: Larger values lead to earlier predictions.')

args = parser.parse_args()
