from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision
"""
    setup basically dataset for training
    dataset, dataloader
"""
batch_size = 4
# transform image to tensor
transformer = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
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

if __name__ == "__main__":
    for i, (inputs, labels) in enumerate(train_loader):
        print(f"Point {i} Input: {inputs.shape} Labels: {labels}")