import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os

# two channels, two classes, index equals number that is assigned to the pixel in the mask
class_names = ['void', 'circle']

def make_image_mask_pair(size: (int, int)) -> (torch.Tensor, torch.Tensor):
    image = torch.zeros((1, *size), dtype=torch.float)
    mask = torch.zeros((1, *size), dtype=torch.float)

    center = torch.rand((2), dtype=torch.float) * torch.tensor(size, dtype=torch.float)
    max_distance = torch.norm(torch.tensor([size[0] / 2, size[1] / 2], dtype=torch.float), p=2)

    for x in range(0, size[0]):
        for y in range(0, size[1]):
            point = torch.tensor([x, y], dtype=torch.float)
            distance = torch.dist(center, point)
            if distance > max_distance:
                image[0, x, y] = 0
            else:
                image[0, x, y] = 1 - distance / max_distance
            
            if distance < max_distance / 2:
                mask[0, x, y] = 1
    
    #plt.imshow(image.squeeze(dim=0))
    #plt.show()
    #plt.imshow(mask.squeeze(dim=0))
    #plt.show()
    return (image, mask)

def generate_dataset_part(count: int, size: (int, int), path: Path):
    if not path.exists():
        os.makedirs(path)

    images_dir = path/'images'
    if not images_dir.exists():
        os.makedirs(images_dir)
    
    masks_dir = path/'masks'
    if not masks_dir.exists():
        os.makedirs(masks_dir)

    for i in range(0, count):
        image_path = images_dir/(str(i) + '.png')
        mask_path = masks_dir/(str(i) + '.png')
        image, mask = make_image_mask_pair(size=size)
        save_image(image, image_path)
        save_image(mask, mask_path)

def generate_mock_dataset(train_count: int, test_count: int, size: (int, int), path: Path):
    if not path.exists():
        os.makedirs(path)

    generate_dataset_part(count=train_count, size=size, path=path/'train')
    generate_dataset_part(count=test_count, size=size, path=path/'test')

    classes_file = path/'classes.txt'
    with open(classes_file, 'w') as file:
        file.write(','.join(class_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate a dataset for testing the unet')
    parser.add_argument('width', type=int, help='the width of the generated image images and masks')
    parser.add_argument('height', type=int, help='the height of the generated image images and masks')
    parser.add_argument('train_count', type=int, help='the number of items to generate for training')
    parser.add_argument('test_count', type=int, help='the number of items to generate for testing')
    parser.add_argument('path', type=str, help='where to store the dataset, dataset structure will be created inside that directory')

    args = parser.parse_args()

    generate_mock_dataset(train_count=args.train_count, test_count=args.test_count, size=(args.width, args.height), path=Path(args.path))