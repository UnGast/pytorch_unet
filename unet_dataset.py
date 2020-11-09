import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from enum import Enum
import PIL
import dataset_generator

class DatasetPart(Enum):
  Train = 'train'
  Test = 'test'

class UNetDataset(Dataset):
  def __init__(self, root_dir: Path, part: DatasetPart, image_extension='png'):
    self.root_dir = root_dir
    self.images_dir = root_dir/part/'images'
    self.masks_dir = root_dir/part/'masks'

    self.classes_file_path = root_dir/'classes.txt'
    with open(self.classes_file_path, 'r') as file:
      self.classes = file.read().split(',')

    self.filenames = [file_path.name for file_path in self.images_dir.glob('*.{}'.format(image_extension))]
    self.filenames = [file_path.name for file_path in self.masks_dir.glob('*.{}'.format(image_extension)) if file_path.name in self.filenames]

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
    mask = transforms.ToTensor()(mask)
    mask = mask[0,:,:].type(torch.LongTensor)
    #mask = nn.functional.one_hot(mask.type(torch.LongTensor)).type(torch.FloatTensor).permute(2, 0, 1)
    return (image, mask)

class TmpUNetDataset(Dataset):
    def __init__(self, item_size: (int, int), item_count: int):
        self.item_size = item_size
        self.item_count = item_count
        
    def __len__(self):
        return self.item_count
    
    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        return make_image_mask_pair(size=self.item_size)

if __name__ == '__main__':
  import sys
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str, help='the root directory of the dataset')
  parser.add_argument('part', type=str, help="'train' or 'test'")
  args = parser.parse_args()
  dataset = UNetDataset(root_dir=Path(args.path), part=args.part)
  print('dataset length', len(dataset))
  print('classes', dataset.classes)
  print('item size', dataset.item_size)
  print('image shape', dataset[0][0].shape)
  print('mask shape', dataset[0][1].shape)