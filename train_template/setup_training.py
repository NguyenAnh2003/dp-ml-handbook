import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision

# setup file
"""
setup loss
setup dataloader
"""

def setup_loss():
    """
    this function used for setup loss function
    """
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn

def setup_dataset():
    """
    setup basically dataset for training
    dataset, dataloader
    """
    batch_size = 16
    # transform image to tensor
    transformer = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))]
    )
    # train set
    training_set = torchvision.datasets.FashionMNIST('', train=True,
                                                     transform=transformer,
                                                     download=True)
    # eval set
    eval_set = torchvision.datasets.FashionMNIST('', train=False,
                                                 transform=transformer,
                                                 download=True)
    # dataloader
    train_loader = DataLoader(training_set, batch_size=batch_size, 
                              shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, 
                             shuffle=False)
    
    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    
    # train_loader, eval_loader, labels
    return train_loader, eval_loader, classes
