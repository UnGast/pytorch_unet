import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path

class UNetDataset(Dataset):
  def __init__(self, root_dir: Path, image_extension='png'):
    self.root_dir = root_dir
    self.images_dir = root_dir/'images'
    self.masks_dir = root_dir/'masks'
    self.filenames = [file_path.name for file_path in self.images_dir.glob('*.{}'.format(image_extension))]
    self.filenames = [file_path.name for file_path in self.masks_dir.glob('*.{}'.format(image_extension)) if file_path.name in self.filenames]

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
  args = parser.parse_args()
  dataset = UNetDataset(root_dir=Path(args.path))
  print('dataset length', len(dataset))