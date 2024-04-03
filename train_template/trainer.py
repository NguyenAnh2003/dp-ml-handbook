import torch
from tqdm import tqdm
from setup_training import *
from data_loader.dataloader import *
import wandb
import time
from train_template.model.neural_model import NeuralModel
from logger.logger_wandb import train_logging
""" train template """

# model setup
model = NeuralModel()
loss_fn = setup_loss() # entropy loss
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # adam optim
# device
device = setup_device()
EPOCHS = 10
best_vloss = 1_000_000

def training_model(exp_name: str):
    """ Validation: It provides an estimate of how well the model is likely to perform on unseen data """
    wandb.init(project="basic_neural", name=exp_name)
    train_losses = []
    eval_losses = [] # epoch val loss
    batch_val_loss = []
    start_time = time.time() # start couting time
    for epoch in range(EPOCHS):
        model.train(True)  # train mode
        # average loss in one epoch
        avg_loss = train_one_epoch(epoch_index=epoch+1,
                                   train_loader=train_loader,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn, model=model)

        train_losses.append(avg_loss) # append avg train loss

        # validation
        running_lossv = 0.0
        # evaluation
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(eval_loader):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_lossv += loss.item()
                batch_val_loss.append(loss.item())

        eval_losses.append(sum(batch_val_loss)/ len(batch_val_loss)) # append dev loss

        # tracking best loss and train loss
        # if avg_loss < best_vloss:
        #     best_vloss = avg_loss

        # print epoch result
        """ epoch result train loss, val loss"""
        # print(f"Epoch: {epoch+1} Train loss: {avg_loss/len(train_loader)}"
        #       f" Dev loss: {running_lossv/len(eval_loader)} \n")
        # WB logging
        wandb.log({"eval_loss/epoch": sum(eval_losses) / len(eval_losses),
                   "train_loss/epoch": avg_loss})  # logging loss per epoch
        
        train_logging(train_loss=avg_loss, dev_loss=sum(eval_losses)/len(eval_losses))

    # end training
    torch.save(model.state_dict(), "./saved_model/nguyenanh.pth")
    # print end of training process
    print(f"End training with epochs: {EPOCHS}\n"
          f"Train loss {min(train_losses)} "
          f"Dev loss {min(eval_losses)} "
          f"Total time: {time.time() - start_time}")

if __name__ == "__main__":
    training_model("4-3")