import numpy as np
import random
from random import sample
import torch
import torchvision.transforms as transforms
from PIL import Image
#Resolution
color_resolution = {'x1': 256, 'x8': 128, 'x64': 64, 'x512': 32}

class ChannelSplit():
    def __init__(self, res, choice, prob=0.5):
        self.res = res
        self.choice = choice
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            img = self._color_global(img, color_resolution[self.res], choice=self.choice)
        return img
    def _color_global(self, image, resolution=128, choice=2):
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        result = []
        for r in range(int(255 / resolution) + 1):
            f_r = np.multiply(image[0], (resolution * r <= image[0]) & ((resolution * (r + 1) - 1) >= image[0]))
            for g in range(int(255 / resolution) + 1):
                f_g = np.multiply(image[1], (resolution * g <= image[1]) & ((resolution * (g + 1) - 1) >= image[1]))
                for b in range(int(255 / resolution) + 1):
                    f_b = np.multiply(image[2], (resolution * b <= image[2]) & ((resolution * (b + 1) - 1) >= image[2]))
                    result.append(np.stack((f_r, f_g, f_b)))
        result = np.array(sample(result, choice), dtype=np.uint8)
        if choice == 1:
            result = np.transpose(np.squeeze(result, axis=0), (1, 2, 0))
        else:
            result = np.transpose(result, (0, 2, 3, 1))
        return result

class ChannelMix():
    def __init__(self, sum=False, prob=0.7, beta=5, width=3):
        self.sum = sum
        self.prob = prob
        self.beta = beta
        self.width = width
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        self.preprocess = preprocess

    def __call__(self, img):
        #H, W, C
        if random.random() < self.prob:
            self.res = 'x64'
            self.choice = 64
            _img = ChannelSplit(res=self.res, choice=self.choice, prob=1.0)(img)

            #B, H, W, C
            dirichlet = np.float32(np.random.dirichlet([1] * self.width))
            beta = np.float32(np.random.beta(self.beta, 1))
            mix = torch.zeros_like(self.preprocess(img))
            #H, W, C
            for i in range(self.width):
                step = int(self.choice / self.width)
                rand = np.random.randint(1, (self.choice + 1) - step*i)
                mixed = _img[np.random.choice(np.arange(0, _img.shape[0]), rand, replace=False)]
                mixed = mixed.sum(axis=0)
                mixed = Image.fromarray(mixed.astype(np.uint8))
                mix += dirichlet[i] * self.preprocess(mixed)
            img = (beta * self.preprocess(img) + (1 - beta) * mix)
        else:
            img = self.preprocess(img)
        return img