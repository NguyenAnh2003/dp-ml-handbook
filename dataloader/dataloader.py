from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from melspectrogram_transforms import melspectrogram_transforms
import yaml
import torch
from torch_load_audio import load_audio

with open('config_melspec.yaml', 'r') as file:
    params = yaml.safe_load(file)

class MyDataset(Dataset):
    def __init__(self, csv, root_dir):
        """
        :param phoneme: csv file
        :param audio_dir: audio directory containing audio file (wav file)
        """
        self.annotations = pd.read_csv(csv)
        self.root_dir = root_dir
    def __getitem__(self, index):
        """
        Dataset convert to melspec and pitch and forward to dataloader
        :param index: index for each audio file
        :return: audio after loading and cannonical correspond to audio
        """
        # path for each index
        audio_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        # audio for each index
        audio = load_audio(audio_path)
        # transform to mel spectrogram - for each index
        melspec = melspectrogram_transforms(audio, params)
        cannonical = self.annotations.iloc[index, 2]
        sample = {"Melspec": melspec, "Cannonical": cannonical} # pair
        return melspec, cannonical

    def __len__(self):
        return len(self.annotations)


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super(MyDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        # pass collate function in here
        self.collate_fn = self.collate_func

    def collate_func(self, batch):
        """
        padding n_frames -> solving too much padding
        1. sort dataset in increasing number of words -> minimizing padding
        2. pass sequential indices
        3. add padding to match dimensions
        """
        melspec, cannonical = batch
        return melspec, cannonical, type(batch)
