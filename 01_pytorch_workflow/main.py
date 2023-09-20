from LinearRegression import LinearRegression
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from preparing_loading_data import X_test, X_train, y_train, y_test
import numpy as np
from pathlib import Path

# Config saving
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "LN.pt"
SAVE_PATH = MODEL_PATH / MODEL_NAME


# torch.manual_seed(42)
model = LinearRegression()
print((model.state_dict()))
print("Set up loss and optim")
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01) # optim
# Training
epoch_count = []
loss_values = []
test_loss_values = []
epochs = 100 # one loop through the data - hyperparameter
for epoch in range(epochs):
    # training mode
    model.train()
    # forward pass
    y_pred = model(X_train)
    # loss func
    loss = loss_fn(y_pred, y_train)
    print(f"Loss: {loss}")
    # Optimizer zero grad
    optimizer.zero_grad()
    # BP
    loss.backward()
    # step optimizer
    optimizer.step()

    # eval mode test set up?
    model.eval()
    with torch.inference_mode(): # turn off grad tracking
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        print(model.state_dict())

plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Test loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

torch.save(model.state_dict(), SAVE_PATH)