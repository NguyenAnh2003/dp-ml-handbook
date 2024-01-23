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
    model.train(True) # train mode
    running_loss = 0.0
    wandb.init(project="base-model")
    for i, batch in tqdm(enumerate(train_loader)):
        # getting input and label
        inputs, labels = batch

        # set inputs and labels to cuda
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # print(inputs.shape)

        # setup grad to zero when getting new data point
        optimizer.zero_grad()
        # make prediction for each batch
        outputs = model(inputs)
        # compute loss, grad
        loss = loss_fn(outputs, labels)
        # gathering and report
        running_loss += loss.item()
        loss.backward()  # cal grad
        # adjust weights
        optimizer.step()

    print(f"Epoch: {epoch_index + 1} Loss: {running_loss / len(train_loader)}")  # logging loss
    wandb.log({"loss": running_loss / len(train_loader)}) # logging loss per epoch
    # avg loss return
    return running_loss
