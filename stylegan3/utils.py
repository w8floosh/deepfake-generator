from typing import Tuple
import PIL
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torch


def normalize_tensor(t: Tensor) -> Tensor:
    min = t.min(dim=1, keepdim=True)[0]
    max = t.max(dim=1, keepdim=True)[0]
    return (t - min) / (max - min + 1e-8)


# def make_transform(translate: Tuple[float, float], angle: float):
#     m = np.eye(3)
#     s = np.sin(angle / 360.0 * np.pi * 2)
#     c = np.cos(angle / 360.0 * np.pi * 2)
#     m[0][0] = c
#     m[0][1] = s
#     m[0][2] = translate[0]
#     m[1][0] = -s
#     m[1][1] = c
#     m[1][2] = translate[1]
#     return m


def print_generated(gen):
    img = (gen[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    plt.imshow(PIL.Image.fromarray(img.cpu().numpy(), "RGB"))
    plt.axis("off")
    plt.show()
