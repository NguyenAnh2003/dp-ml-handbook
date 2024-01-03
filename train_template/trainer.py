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
EPOCHS = 50
def training_model():
    train_losses = []
    eval_losses = []
    for epoch in range(EPOCHS):
        # average loss in one epoch
        model.train(True)
        # training
        avg_loss = train_one_epoch(epoch_index=epoch,
                                   train_loader=train_loader,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn, model=model)
        train_losses.append(avg_loss) # append avg train loss
        # evaluation
        model.eval()
        with torch.no_grad():
            running_lossv = 0
            for i, batch in enumerate(eval_loader):
                inputs, labels = batch
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_lossv += loss.item()

        # print epoch result
        """ epoch result train loss, val loss"""
        print(f"Train loss: {avg_loss/len(train_loader)}"
              f" Dev loss: {running_lossv/len(eval_loader)} \n")

    # print end of training process
    print(f"End training with epochs: {EPOCHS}\n"
          f"Train loss {max(train_losses)}"
          f"Dev loss {max(eval_losses)}")