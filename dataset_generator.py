import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
from multiprocessing import Pool
import random
from PIL import Image

# two channels, two classes, index equals number that is assigned to the pixel in the mask
class_names = ['void', 'circle']

def make_simple_image_mask_pair(size: (int, int)) -> (torch.Tensor, torch.Tensor):
    image = torch.zeros((1, *size), dtype=torch.int)
    mask = torch.zeros((1, *size), dtype=torch.int)

    center = torch.rand((2), dtype=torch.float) * torch.tensor(size, dtype=torch.float)
    max_distance = torch.norm(torch.tensor([size[0] / 2, size[1] / 2], dtype=torch.float), p=2) / 2

    for x in range(0, size[0]):
        for y in range(0, size[1]):
            point = torch.tensor([x, y], dtype=torch.float)
            distance = torch.dist(center, point)
            if distance < max_distance:
                image[0, x, y] = 255
                mask[0, x, y] = 1

    return (image, mask)

def make_advanced_image_mask_pair(size: (int, int)) -> (torch.Tensor, torch.Tensor):
    image = torch.zeros((3, *size), dtype=torch.int)
    mask = torch.zeros((1, *size), dtype=torch.int)

    center = torch.rand((2), dtype=torch.float) * torch.tensor(size, dtype=torch.float)
    max_distance = torch.norm(torch.tensor([size[0] / 2, size[1] / 2], dtype=torch.float), p=2) / 2

    channel = random.randint(0, 2)
    
    for x in range(0, size[0]):
        for y in range(0, size[1]):
            point = torch.tensor([x, y], dtype=torch.float)
            distance = torch.dist(center, point)
            if distance < max_distance:
                image[channel, x, y] = 255
                mask[0, x, y] = 1

    return (image, mask)

def make_image_mask_pair(difficulty: str, size: (int, int)) -> (torch.Tensor, torch.Tensor):
    if difficulty == 'simple':
        return make_simple_image_mask_pair(size=size)
    elif difficulty == 'advanced':
        return make_advanced_image_mask_pair(size=size)
    else:
        raise Exception('unsupported difficulty provided to dataset generation')

def poolable_write_item(index: int, difficulty: str, size: (int, int), images_dir: Path, masks_dir: Path):
    image_path = images_dir/(str(index) + '.png')
    mask_path = masks_dir/(str(index) + '.png')
    image, mask = make_image_mask_pair(difficulty=difficulty, size=size)
    image = Image.fromarray(image.numpy().astype('uint8').transpose(2, 1, 0).squeeze())
    mask = Image.fromarray(mask.numpy().astype('uint8').transpose(2, 1, 0).squeeze())
    image.save(image_path)
    mask.save(mask_path)

def generate_dataset_part(difficulty: str, count: int, size: (int, int), path: Path):
    if not path.exists():
        os.makedirs(path)

    images_dir = path/'images'
    if not images_dir.exists():
        os.makedirs(images_dir)
    
    masks_dir = path/'masks'
    if not masks_dir.exists():
        os.makedirs(masks_dir)

    with Pool(5) as p:
        p.starmap(poolable_write_item, [(index, difficulty, size, images_dir, masks_dir) for index in range(0, count)])

def generate_mock_dataset(difficulty: str, train_count: int, valid_count: int, size: (int, int), path: Path):
    if not path.exists():
        os.makedirs(path)

    generate_dataset_part(difficulty=difficulty, count=train_count, size=size, path=path/'train')
    generate_dataset_part(difficulty=difficulty, count=valid_count, size=size, path=path/'valid')

    classes_file = path/'classes.csv'
    with open(classes_file, 'w') as file:
        file.write(','.join(class_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate a dataset for testing the unet')
    parser.add_argument('difficulty', type=str, choices=['simple', 'advanced'], help='how difficult should the generated images be to mask')
    parser.add_argument('width', type=int, help='the width of the generated image images and masks')
    parser.add_argument('height', type=int, help='the height of the generated image images and masks')
    parser.add_argument('train_count', type=int, help='the number of items to generate for training')
    parser.add_argument('valid_count', type=int, help='the number of items to generate for testing')
    parser.add_argument('path', type=str, help='where to store the dataset, dataset structure will be created inside that directory')

    args = parser.parse_args()

    generate_mock_dataset(difficulty=args.difficulty, train_count=args.train_count, valid_count=args.valid_count, size=(args.width, args.height), path=Path(args.path))