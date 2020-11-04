import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path
from enum import Enum

class DatasetPart(Enum):
  Train = 'train'
  Test = 'test'

class UNetDataset(Dataset):
  def __init__(self, root_dir: Path, part: DatasetPart, image_extension='png'):
    self.root_dir = root_dir
    self.images_dir = root_dir/part/'images'
    self.masks_dir = root_dir/part/'masks'
    self.filenames = [file_path.name for file_path in self.images_dir.glob('*.{}'.format(image_extension))]
    self.filenames = [file_path.name for file_path in self.masks_dir.glob('*.{}'.format(image_extension)) if file_path.name in self.filenames]

    if len(self.filenames) > 0:
      image, _ = self[0]
      self.item_size = (image.shape[1], image.shape[2])

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
    image = read_image(str(self.images_dir/self.filenames[index]))
    mask = read_image(str(self.masks_dir/self.filenames[index]))
    mask = mask[0,:,:].unsqueeze(axis=0)
    return (image, mask)

if __name__ == '__main__':
  import sys
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str, help='the root directory of the dataset')
  parser.add_argument('part', type=str, help="'train' or 'test'")
  args = parser.parse_args()
  dataset = UNetDataset(root_dir=Path(args.path), part=args.part)
  print('dataset length', len(dataset))
  print('item size', dataset.item_size)