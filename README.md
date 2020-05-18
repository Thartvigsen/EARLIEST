# EARLIEST
PyTorch code for "Adaptively-Halting Policy Network for Early Classification", published at KDD'19.
EARLIEST is currently only available in PyTorch.

Introductory use:
```python
from model import EARLIEST

# --- hyperparameters ---
N_FEATURES = 5 # Number of variables in your data (if we have clinical time series recording both heart rate and blood pressure, this would be a 2-dimensional time series, regardless of the number of timesteps)
N_CLASSES = 3 # Number of classes
HIDDEN_DIM = 50 # Hidden dimension of the RNN
CELL_TYPE = "LSTM" # Use an LSTM as the Recurrent Memory Cell.

# --- defining data and model ---
d = torch.rand((5, 1, N_FEATURES)) # A simple synthetic time series.
labels = torch.tensor([0], dtype=torch.long) # A simple synthetic label.
m = EARLIEST(N_FEATURES, N_CLASSES, HIDDEN_DIM, CELL_TYPE) # Initializing the model

# --- inference ---
# Now we can use m for inference
y_hat = m(d)

# --- computing loss and gradients ---
# Computing the loss is quite simple:
loss = m.applyLoss(y_hat, labels)
loss.backward() # Compute all gradients
```

For a more comprehensive example of training EARLIEST on a very simple dataset, please investigate [this example](univariate_example.py).
You can simply run the file:
```bash
python univariate_example.py
```
**Requirements**: PyTorch 1.0+, NumPy, and Scikit-Learn.

In practice, it may prove helpful to:
1. Alter the epsilon values over time -- you can tune this to change how much you want the model to randomly wait/halt: Early on in training, waiting helps the classifier cover the discrimination space better.
2. Apply the Discriminator and let the Controller look at the predictions made
   prior to making its decision. This is a different model but would likely lead
   to better classification!
If you try one of these, please let Tom know how it goes!

If you find this code helpful, feel free to cite our paper:
```
@inproceedings{hartvigsen2019adaptive,
  title={Adaptive-Halting Policty Network for Early Classification},
  author={Hartvigsen, Thomas and Sen, Cansu and Kong, Xiangnan and Rundensteiner, Elke},
  booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={101--110},
  year={2019},
  organization={ACM}
}
```
