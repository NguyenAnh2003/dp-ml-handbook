from model.neural_model import *
import torch

if __name__ == "__main__":
    model = NeuralModel()
    model.load_state_dict(torch.load("./saved_model/nguyenanh.pth"))
    print(f"Model params: {sum([p.nelement() for p in model.parameters()])} "
          f"Model shape: {model.share_memory()}")