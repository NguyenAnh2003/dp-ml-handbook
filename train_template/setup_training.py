import torch.cuda
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import wandb

# setup file
"""
setup loss
setup dataloader
setup device
"""

def setup_device():
    device = torch.cuda if torch.cuda.is_available() else "cpu"
    return device

def setup_loss():
    """
    this function used for setup loss function
    """
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn

def train_one_epoch(epoch_index, train_loader,
                    optimizer, loss_fn, model):
    running_loss = 0
    last_loss = 0
    for i, batch in tqdm(enumerate(train_loader)):
        # getting input and label
        inputs, labels = batch
        # setup grad to zero when getting new data point
        optimizer.zero_grad()
        # make prediction for each batch
        outputs = model(inputs)
        # compute loss, grad
        loss = loss_fn(outputs, labels)
        loss.backward()
        # adjust weights
        optimizer.step()
        # gathering and report
        running_loss += loss.item()
        print(f"Batch:{i} Loss:{running_loss}") # logging loss
        # wandb.log({"loss": running_loss})
    # avg loss return
    return last_loss