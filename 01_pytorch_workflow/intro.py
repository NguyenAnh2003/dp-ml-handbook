import torch
# All of pytorch building blocks
from torch import nn
import matplotlib.pyplot as plt
from print_pure import print_pure

flow = {1: "data (prepare and load)",
                      2: "build model",
                      3: "fitting the model to data (training)",
                      4: "making predictions and evaluating a model (inference)",
                      5: "saving and loading a mode",
                      6: "put all together"}

print_pure(torch.__version__)

