import torch
import torch.nn as nn

output = torch.randn(4, 10)
print(output, 'size', output.shape)
labels = torch.tensor([1, 5, 3, 7])
# loss
lossfn = nn.CrossEntropyLoss()

loss = lossfn(output, labels)
print('LOSS: {}'.format(loss.item()))