import torch
from tqdm import tqdm
from setup_training import *
from data_loader.dataloader import *
from train_template.model.neural_model import NeuralModel
""" train template """

# model setup
model = NeuralModel()
loss_fn = setup_loss() # entropy loss
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # adam optim
# device
device = setup_device()
EPOCHS = 10
best_vloss = 1_000_000

def training_model():
    train_losses = []
    eval_losses = []
    for epoch in range(EPOCHS):
        # average loss in one epoch
        # training
        avg_loss = train_one_epoch(epoch_index=epoch,
                                   train_loader=train_loader,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn, model=model)

        train_losses.append(avg_loss/len(train_loader)) # append avg train loss

        # validation
        running_lossv = 0.0
        # evaluation
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(eval_loader):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_lossv += loss.item()

        eval_losses.append(running_lossv/ len(eval_loader)) # append dev loss

        # tracking best loss and train loss
        # if avg_loss < best_vloss:
        #     best_vloss = avg_loss

        # print epoch result
        """ epoch result train loss, val loss"""
        print(f"Epoch: {epoch+1} Train loss: {avg_loss/len(train_loader)}"
              f" Dev loss: {running_lossv/len(eval_loader)} \n")

    # end training
    torch.save(model.state_dict(), "./saved_model/nguyenanh.pth")
    # print end of training process
    print(f"End training with epochs: {EPOCHS}\n"
          f"Train loss {max(train_losses)}"
          f"Dev loss {max(eval_losses)}")

if __name__ == "__main__":
    training_model()