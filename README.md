# Adaptive-Halting Policy Network for Early Classification
This repository is the official implementation of [Adaptive-Halting Policy Network for Early Classification](https://dl.acm.org/doi/10.1145/3394486.3403191?cid=99659453882), published at ACM SIGKDD 2019.

## Training
To train the model on our simple synthetic dataset, run this command:
```
python train.py --dataset synthetic
```
Our code include arguments for different hyperparameters of the model and dataset which may also be passed into the training file at runtime.
This code will train a model, evaluate it on a validation split of the training
set, then save the model.

## Testing
To test the model on the synthetic dataset, run this command:
```
python test.py --dataset synthetic
```
This requires that all model architecture parameters are kept the same but will
load in the saved model. If desired, the path to the proper saved model may be
passed in as an additional argument. This code will load the saved model and run
it once on the testing set. Then, it will print out the testing accuracy and the
average halting time.

## General Example
Integrating the EARLIEST model with existing PyTorch code is easy:
```python
from model import EARLIEST

# --- hyperparameters ---
N_FEATURES = 5 # Number of variables in your data (if we have clinical time series recording both heart rate and blood pressure, this would be a 2-dimensional time series, regardless of the number of timesteps)
N_CLASSES = 3 # Number of classes
HIDDEN_DIM = 50 # Hidden dimension of the RNN
CELL_TYPE = "LSTM" # Use an LSTM as the Recurrent Memory Cell.
NUM_TIMESTEPS = 10 # Number of timesteps in your input series (EARLIEST doesn't need this as input, this is just set to create synthetic series)
BATCH_SIZE = 32 # Pick your batch size
LAMBDA = 0.0 # Set lambda, the emphasis on earliness

# --- defining data and model ---
d = torch.rand((NUM_TIMESTEPS, BATCH_SIZE, N_FEATURES)) # A simple synthetic time series.
labels = torch.randint(2, size=(BATCH_SIZE)) # Random synthetic labels.
m = EARLIEST(N_FEATURES, N_CLASSES, HIDDEN_DIM, CELL_TYPE, lam=LAMBDA) # Initializing the model

# --- inference ---
# Now we can use m for inference
logits, halting_points = m(d)
_, predictions = torch.max(torch.softmax(logits, 1), 1)

# --- computing loss and gradients ---
# Computing the loss is quite simple:
loss = m.applyLoss(logits, labels)
loss.backward() # Compute all gradients
```

In practice, it may prove helpful to:
1. Alter the epsilon values over time -- you can tune this to change how much you want the model to randomly wait/halt: Early on in training, waiting helps the classifier cover the discrimination space better.
2. Apply the Discriminator and let the Controller look at the predictions made
   prior to making its decision. This is a different model but would likely lead
   to better classification!
If you try one of these, please let Tom know how it goes!

**Requirements**: PyTorch 1.0+, NumPy, and Scikit-Learn.

If you find this code helpful, feel free to cite our paper:
```
@inproceedings{hartvigsen2019adaptive,
  title={Adaptive-Halting Policy Network for Early Classification},
  author={Hartvigsen, Thomas and Sen, Cansu and Kong, Xiangnan and Rundensteiner, Elke},
  booktitle={Proceedings of the 25th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={101--110},
  year={2019},
  organization={ACM}
}
```
