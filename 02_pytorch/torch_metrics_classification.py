from torchmetrics import Accuracy
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# set up metric
metric_acc = Accuracy().to(device)
# Cal
x = torch.randn()
