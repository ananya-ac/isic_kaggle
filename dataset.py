import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset


class ISIC_Dataset(Dataset):
    def __init__(self, data_path, dataframe, transform=None):
        self.data_path = data_path
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.data_path, self.dataframe['isic_id'].iloc[idx] + '.jpg')
        
        image = (np.array(Image.open(img_name).convert('RGB')) / 255)
        target = self.dataframe['target'].iloc[idx]
        
        if self.transform:
            image = self.transform(image=image)['image'].transpose(2, 0, 1)
            
        return torch.tensor(image, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

