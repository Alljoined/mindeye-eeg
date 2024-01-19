from torch.utils.data import Dataset
import numpy as np
import os
import torch
from scipy.interpolate import interp1d
from PIL import Image
import numpy as np
from transformers import AutoProcessor

def identity(x):
    return x

#----------------------------------------------------------------------------

EEG_EXTENSIONS = [
    '.mat'
]

class EEGDataset(Dataset):
    
    # Constructor
    def __init__(self, eeg_signals_path, image_transform=identity, subject = 4):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        # else:
        # print(loaded)
        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject]
        else:
            self.data = loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = '/workspace/eeg_reconstruction/shared/brain2image/imageNet_images/'
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        # Compute size
        self.size = len(self.data)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        # print(self.data[i])
        eeg = self.data[i]["eeg"].float().t()

        eeg = eeg[20:460,:]
        ##### 2023 2 13 add preprocess and transpose
        eeg = np.array(eeg.transpose(0,1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()
        ##### 2023 2 13 add preprocess
        label = torch.tensor(self.data[i]["label"]).long()

        # Get label
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name+'.JPEG')
        # print(image_path)
        image_raw = Image.open(image_path).convert('RGB') 
        
        image = np.array(image_raw) / 255.0
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}


class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)

        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if i <= len(self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size

        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


def create_EEG_dataset(eeg_signals_path='../DreamDiffusion/datasets/eeg_5_95_std.pth', 
            splits_path = '../DreamDiffusion/datasets/block_splits_by_image_single.pth',
            # splits_path = '../DreamDiffusion/datasets/block_splits_by_image_all.pth',
            image_transform=identity, subject = 0):
    # if subject == 0:
        # splits_path = '../DreamDiffusion/datasets/block_splits_by_image_all.pth'
    if isinstance(image_transform, list):
        dataset_train = EEGDataset(eeg_signals_path, image_transform[0], subject )
        dataset_test = EEGDataset(eeg_signals_path, image_transform[1], subject)
    else:
        dataset_train = EEGDataset(eeg_signals_path, image_transform, subject)
        dataset_test = EEGDataset(eeg_signals_path, image_transform, subject)
    split_train = Splitter(dataset_train, split_path = splits_path, split_num = 0, split_name = 'train')
    split_test = Splitter(dataset_test, split_path = splits_path, split_num = 0, split_name = 'test')
    return (split_train, split_test)
