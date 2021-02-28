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
  def __init__(self, item_paths: [(Path, Path)], classes: [str], transforms):
    """
    a dataset where items are of shape (image: torch.Tensor, mask: torch.Tensor),

    transforms should be given as: [(apply_on_y: bool, transform)]
    """

    self.item_paths = item_paths

    self.classes = classes

    self._transforms = transforms
    self._transformed_per_original = 0
    self.transforms_for_items = []

    self.update_transforms_for_items()

    self.mask_channels = 1
    if len(self.item_paths) > 0:
      image, _ = self[0]
      self.image_channels = image.shape[0]
      self.item_size = (image.shape[1], image.shape[2])
    
  @classmethod
  def from_directory(cls, directory: Path, inputs_directory_name: str='images', targets_directory_name: str='masks', classes_file_path=None, image_extension='png', mask_extension='png', transforms=[]) -> 'SegmentationDataset':
    input_paths = [file_path for file_path in (directory/inputs_directory_name).glob('*.{}'.format(image_extension))]
    item_paths = []
    for input_path in input_paths:
      target_path = directory/targets_directory_name/'{}.{}'.format(input_path.stem, mask_extension)
      if target_path.exists():
        item_paths.append((input_path, target_path))

    if classes_file_path is None:
      classes_file_path = directory/'classes.csv'
    classes = []
    with open(classes_file_path, 'r') as file:
      classes = file.read().split(',')

    dataset = cls(item_paths=item_paths, classes=classes, transforms=transforms)

    return dataset

  def split_by_percentage(self, percentage: float) -> ('SegmentationDataset', 'SegmentationDataset'):
    """
    Split this dataset into two parts (for training and validation) by the given percentage.
    The split should always be the same if the items were read in the same order.
    """
    if percentage <= 1:
      percentage *= 100

    split_index = math.ceil(len(self) * percentage / 100)
    item_paths1 = list(self.item_paths[:split_index])
    item_paths2 = []
    if split_index < len(self):
      item_paths2 = list(self.item_paths[split_index:])

    dataset1 = SegmentationDataset(item_paths=item_paths1, classes=self.classes, transforms=self.transforms)
    dataset2 = SegmentationDataset(item_paths=item_paths2, classes=self.classes, transforms=self.transforms)

    return (dataset1, dataset2)

  def __len__(self):
    return len(self.item_paths) * (self.transformed_per_original + 1)

  @property
  def transforms(self):
    return self._transforms
  
  @transforms.setter
  def transforms(self, new_value):
    self._transforms = new_value
    self.update_transforms_for_items()

  @property
  def transformed_per_original(self):
    return self._transformed_per_original
  
  @transformed_per_original.setter
  def transformed_per_original(self, new_value):
    self._transformed_per_original = new_value
    self.update_transforms_for_items()

  def update_transforms_for_items(self):
    self.transforms_for_items = [[]] * len(self)
    for index in range(len(self)):
      if index % (self.transformed_per_original + 1) == 0:
        self.transforms_for_items[index] = []
      else:
        self.transforms_for_items[index] = self.get_random_transforms()

  def get_random_transforms(self):
    return random.sample(self.transforms, random.randint(0, len(self.transforms)))

  def get_image_as_tensor(self, path) -> torch.Tensor:
    image = PIL.Image.open(path)
    tensor = transforms.ToTensor()(image)
    return tensor 

  def get_input(self, path, transforms) -> torch.Tensor:
    original = self.get_image_as_tensor(path)
    transformed = original
    if len(transforms) > 0:
      transforms = [x[1] for x in transforms]
      transformed = torch.nn.Sequential(*transforms)(transformed)
    return transformed
  
  def get_target(self, path, transforms) -> torch.Tensor:
    original = (self.get_image_as_tensor(path).squeeze(dim=0) * 255).type(torch.LongTensor)
    filtered_transforms = [x[1] for x in transforms if x[0]]
    transformed = original
    if len(filtered_transforms) > 0:
      transformed = torch.nn.Sequential(*filtered_transforms)(transformed.unsqueeze(dim=0)).squeeze(dim=0)
    return transformed

  def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
    transforms = self.transforms_for_items[index] or []
    input_path, target_path = self.item_paths[int(index / (self.transformed_per_original + 1))]
    input = self.get_input(input_path, transforms)
    target = self.get_target(target_path, transforms)
    return (input, target)

  def show_item(self, index, figsize=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(self[index][0].permute(1, 2, 0))
    axes[1].imshow(self[index][1], c, cmap='hsv')
    fig.show()

  def show_items(self, n_items=1, items_per_row=3, figsize=None):
    row_count = math.ceil(n_items / items_per_row)
    column_count = items_per_row * 2
    if row_count == 1:
      column_count = min(column_count, n_items * 2)
    
    fig, axes = plt.subplots(row_count, column_count, squeeze=False, figsize=figsize)

    for i in range(0, n_items):
      row = math.floor(i / items_per_row)
      column = i % items_per_row * 2
      input = self[i][0]
      input = input.permute(1, 2, 0)
      merged_input = input[:,:,:3]
      for i in range(3, input.shape[2]):
        merge_channel_index = i % 3
        merge_channel_data = input[:,:,i]
        merge_channel_data = (merge_channel_data - merge_channel_data.min()) / (merge_channel_data.max() - merge_channel_data.min())
        merged_input[:,:,merge_channel_index] += merge_channel_data
      axes[row, column].imshow(merged_input)
      axes[row, column+1].imshow(self[i][1], cmap='hsv')

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