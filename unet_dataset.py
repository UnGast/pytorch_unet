import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
from enum import Enum
import PIL
from . import dataset_generator
import matplotlib.pyplot as plt
import math

class DatasetPart(Enum):
  Train = 'train'
  Valid = 'valid'

class UNetDataset(Dataset):
  def __init__(self, root_dir: Path, part: DatasetPart, image_extension='png', mask_extension='png'):
    self.root_dir = root_dir
    self.images_dir = root_dir/part/'images'
    self.masks_dir = root_dir/part/'masks'

    self.classes_file_path = root_dir/'classes.csv'
    with open(self.classes_file_path, 'r') as file:
      self.classes = file.read().split(',')

    self.filenames = [file_path.name for file_path in self.images_dir.glob('*.{}'.format(image_extension))]
    self.filenames = [file_path.name for file_path in self.masks_dir.glob('*.{}'.format(mask_extension)) if file_path.name in self.filenames]

    self.image_channels = 3
    self.mask_channels = 1
    if len(self.filenames) > 0:
      image, _ = self[0]
      self.item_size = (image.shape[2], image.shape[1])

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
    image = PIL.Image.open(str(self.images_dir/self.filenames[index]))
    image = transforms.ToTensor()(image)
    mask = PIL.Image.open(str(self.masks_dir/self.filenames[index]))
    mask = torch.from_numpy(np.array(mask)).squeeze().type(torch.FloatTensor)
    return (image, mask)

  def show_item(self, index, figsize=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(self[index][0].permute(1, 2, 0))
    axes[1].imshow(self[index][1])
    fig.show()

  def show_items(self, n_items=1, items_per_row=3, figsize=None):
    row_count = math.ceil(n_items / items_per_row)
    column_count = items_per_row * 2
    
    fig, axes = plt.subplots(row_count, column_count, squeeze=False, figsize=figsize)

    for i in range(0, n_items):
      row = math.floor(i / items_per_row)
      column = i % items_per_row * 2
      axes[row, column].imshow(self[i][0].permute(1, 2, 0))
      axes[row, column+1].imshow(self[i][1])

    fig.show()
    

class TmpUNetDataset(Dataset):
    def __init__(self, item_size: (int, int), item_count: int):
        self.item_size = item_size
        self.item_count = item_count
        self.items = [make_image_mask_pair(size=self.item_size) for _ in range(0, item_count)]
        
    def __len__(self):
        return self.item_count
    
    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        return self.items

if __name__ == '__main__':
  import sys
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str, help='the root directory of the dataset')
  parser.add_argument('part', type=str, choices=[DatasetPart.Train, DatasetPart.Valid])
  args = parser.parse_args()
  dataset = UNetDataset(root_dir=Path(args.path), part=args.part)
  print('dataset length', len(dataset))
  print('classes', dataset.classes)
  print('item size', dataset.item_size)
  print('image shape', dataset[0][0].shape)
  print('mask shape', dataset[0][1].shape)