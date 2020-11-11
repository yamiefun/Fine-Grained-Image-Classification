from torch.utils.data.dataset import Dataset
from torchvision import transforms as trns
from torchvision import utils
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import os
import PIL.Image as Image
import csv

# create corresponding ids for all labels in training dataset
df = pd.read_csv("training_labels.csv")
label = df['label']
files = df['id']
files = [str(file).zfill(6) for file in files]
files = [file+".jpg" for file in files]
files = [os.path.join("./training_data/training_data", file) for file in files]

target_label = []
label_dict = pd.read_csv("label_dict.csv",
                         header=None, index_col=0).to_dict()[1]
for lbl in label:
    target_label.append(label_dict[lbl])

# trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# trns.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_transform = trns.Compose([
    trns.Resize((512, 512)),
    trns.RandomCrop((448, 448)),
    trns.RandomHorizontalFlip(),
    trns.ColorJitter(brightness=0.126, saturation=0.5),
    trns.ToTensor(),
    trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# dataloader for training set
def default_loader(path):
    img_pil = Image.open(path).convert('RGB')
    img_tnsr = train_transform(img_pil)
    return img_tnsr


# training dataset
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images = files
        self.target = target_label
        self.loader = loader

    def __getitem__(self, index):
        file_name = self.images[index]
        img = self.loader(file_name)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)
