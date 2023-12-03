import torch.utils.data

from dataloader import MyDataset, MyDataLoader

dataset = MyDataset(csv='train_phones.csv', root_dir='VMD-VLSP23-training set/')
# set up train, eval size
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
# dataset
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
# dataloader
train_loader = MyDataLoader(train_set, batch_size=4, num_workers=4)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    # for batch_idx, (melspecs, cannonicals) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}")
    #     print("Inputs: ", melspecs)
    #     print("Targets: ", cannonicals)

    for i in train_loader:
        print("Index: ", i)
