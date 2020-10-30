import torch
import matplotlib.pyplot as plt

def make_input_mask_pair(size: (int, int)) -> (torch.Tensor, torch.Tensor):
    input = torch.zeros((1, *size), dtype=torch.float)
    mask = torch.zeros((1, *size), dtype=torch.float)

    center = torch.rand((2), dtype=torch.float) * torch.tensor(size, dtype=torch.float)
    max_distance = torch.norm(torch.tensor([size[0] / 2, size[1] / 2], dtype=torch.float), p=2)

    for x in range(0, size[0]):
        for y in range(0, size[1]):
            point = torch.tensor([x, y], dtype=torch.float)
            distance = torch.dist(center, point)
            if distance > max_distance:
                input[0, x, y] = 0
            else:
                input[0, x, y] = 1 - distance / max_distance
            
            if distance < max_distance / 2:
                mask[0, x, y] = 1
    
    plt.imshow(input.squeeze(dim=0))
    plt.show()
    plt.imshow(mask.squeeze(dim=0))
    plt.show()

def generate_mock_data(count: int, size: (int, int)) -> (torch.Tensor, torch.Tensor):
    for _ in range(0, count):
        pass

make_input_mask_pair(size=(800, 800))