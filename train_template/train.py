import torch
import torch.nn as nn
import torch.optim as optim
from setup_training import *
from neural_model import NeuralModel
""" train template """

# model setup
model = NeuralModel()
loss_fn = setup_loss() # entropy loss
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # adam optim
train_loader, eval_loader, labels = setup_dataset()

print(train_loader)

def train_one_epoch(epoch_index):
    running_loss = 0
    last_loss = 0
    for i, batch in enumerate(train_loader):
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
        print(f"Batch:{i} Loss:{running_loss}")
    return last_loss

for epoch in range(50):
    # average loss in one epoch
    model.train(True)
    avg_loss = train_one_epoch(epoch_index=epoch)
    
    # evaluation
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            inputs, labels = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            


