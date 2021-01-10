import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
from enum import Enum
import PIL
import matplotlib.pyplot as plt
import math
import random

class DatasetPart(Enum):
  Train = 'train'
  Valid = 'valid'

class SegmentationDataset(Dataset):
  """
  a dataset where items are of shape (image: torch.Tensor, mask: torch.Tensor)
  """
  def __init__(self, root_dir: Path, part: DatasetPart, image_extension='png', mask_extension='png', transforms=[]):
    self.root_dir = root_dir
    self.images_dir = root_dir/part/'images'
    self.masks_dir = root_dir/part/'masks'

    self.classes_file_path = root_dir/'classes.csv'
    with open(self.classes_file_path, 'r') as file:
      self.classes = file.read().split(',')

    self.filenames = [file_path.name for file_path in self.images_dir.glob('*.{}'.format(image_extension))]
    self.filenames = [file_path.name for file_path in self.masks_dir.glob('*.{}'.format(mask_extension)) if file_path.name in self.filenames]

    self.transforms = transforms

    self.mask_channels = 1
    if len(self.filenames) > 0:
      image, _ = self[0]
      self.image_channels = image.shape[0]
      self.item_size = (image.shape[2], image.shape[1])
  
  def __len__(self):
    return len(self.filenames)

  def transform(self, partial: torch.Tensor):
    for transform in self.transforms:
      partial = transform(partial)
    return partial

  def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
    image = PIL.Image.open(str(self.images_dir/self.filenames[index]))
    image = self.transform(image)
    image = transforms.ToTensor()(image)
    mask = PIL.Image.open(str(self.masks_dir/self.filenames[index]))
    mask = self.transform(mask)
    mask = torch.from_numpy(np.array(mask)).squeeze().type(torch.LongTensor) # use this approach to prevent transforms.ToTensor converting everything to floats
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
    
class SegmentationTestDataset(SegmentationDataset):
    def __init__(self, item_size: (int, int), item_count: int):
        self.item_size = item_size
        self.item_count = item_count
        self.classes = ['nothing', 'something']
        self.image_channels = 3 
        self.items = [self.generate_item() for _ in range(0, item_count)]

    def generate_item(self) -> (torch.Tensor, torch.Tensor):
      image = torch.zeros((self.image_channels, *self.item_size), dtype=torch.float)
      mask = torch.zeros((*self.item_size), dtype=torch.long)

      center = torch.rand((2), dtype=torch.float) * torch.tensor(self.item_size, dtype=torch.float)
      max_distance = torch.norm(torch.tensor([self.item_size[0] / 2, self.item_size[1] / 2], dtype=torch.float), p=2) / 2

      channel = random.randint(0, 2)
      
      for x in range(0, self.item_size[0]):
          for y in range(0, self.item_size[1]):
              point = torch.tensor([x, y], dtype=torch.float)
              distance = torch.dist(center, point)
              if distance < max_distance:
                  image[channel, x, y] = 255
                  mask[x, y] = 1

      return (image, mask)

    def __len__(self):
        return self.item_count
    
    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        return self.items[index]