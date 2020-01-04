import numpy as np
import random
from random import sample
import torch
import torchvision.transforms as transforms
from PIL import Image
#Resolution
color_resolution = {'x1': 256, 'x8': 128, 'x64': 64, 'x512': 32}
color_resolution_v2 = {'x1': 256, 'x2': 128, 'x4': 64, 'x8': 32, 'x16': 16}

class ChannelSplit():
    def __init__(self, res, choice, skip=False, prob=0.5):
        self.res = res
        self.choice = choice
        self.skip = skip
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            img = self._color_global(img, color_resolution[self.res], choice=self.choice, skip=self.skip)
        return img
    def _color_global(self, image, resolution=128, choice=2, skip=False):
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        _skip = 0
        if resolution == 256:
            _skip = -2
        result = []
        for r in range(int(255 / resolution) + 1):
            f_r = np.multiply(image[0], (resolution * r <= image[0]) & ((resolution * (r + 1) - 1) >= image[0]))
            for g in range(int(255 / resolution) + 1):
                f_g = np.multiply(image[1], (resolution * g <= image[1]) & ((resolution * (g + 1) - 1) >= image[1]))
                for b in range(int(255 / resolution) + 1):
                    _skip += 1
                    if skip and (_skip == 1 or _skip == (int(255 / resolution) + 1) ** 3):
                        continue
                    f_b = np.multiply(image[2], (resolution * b <= image[2]) & ((resolution * (b + 1) - 1) >= image[2]))
                    result.append(np.stack((f_r, f_g, f_b)))
        result = np.array(sample(result, choice), dtype=np.uint8)
        if choice == 1:
            result = np.transpose(np.squeeze(result, axis=0), (1, 2, 0))
        else:
            result = np.transpose(result, (0, 2, 3, 1))
        return result

class ChannelSplit2():
    def __init__(self, res, choice, skip=False, sum=True, prob=0.5):
        self.res = res
        self.choice = choice
        self.skip = skip
        self.sum = sum
        self.prob = prob
    def __call__(self, img):
        if random.random() < self.prob:
            img = self._color_global2(img, color_resolution_v2[self.res], choice=self.choice, skip=self.skip, sum=self.sum)
        return img
    def _color_global2(self, image, resolution=128, choice=2, skip=False, sum=False):
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        _skip = 0
        if resolution == 256 or resolution == 128:
            _skip = -2
        result = []
        for c in range(int(255 / resolution) + 1):
            _skip += 1
            if skip and (_skip == 1 or _skip == (int(255 / resolution) + 1)):
                continue
            result.append(np.multiply(image, (resolution * c <= image) & ((resolution * (c + 1) - 1) >= image)))
        result = np.array(sample(result, choice))

        if choice == 1:
            result = np.transpose(np.squeeze(result, axis=0), (1, 2, 0))
        else:
            if sum:
                result = result.sum(axis=0)
                result = np.transpose(result, (1, 2, 0)).astype(np.uint8)
            else:
                result = np.transpose(result, (0, 2, 3, 1))
        return result

class ChannelMix():
    def __init__(self, skip=False, sum=False, prob=0.5, beta=10, width=3):
        #self.res = res
        #self.choice = choice
        self.skip = skip
        self.sum = sum
        self.prob = prob
        self.beta = beta
        self.width = width

    def __call__(self, img):
        #H, W, C
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        if random.random() < self.prob:
            if random.random() < 0.5:
                self.res = 'x64'
                self.choice = 64
                _img = ChannelSplit(res=self.res, choice=self.choice, skip=self.skip, prob=1.0)(img)
            else:
                self.res = 'x8'
                self.choice = 8
                _img = ChannelSplit2(res=self.res, choice=self.choice, skip=self.skip, sum=self.sum, prob=1.0)(img)

            #B, H, W, C
            dirichlet = np.float32(np.random.dirichlet([1] * self.width))
            beta = np.float32(np.random.beta(self.beta, 1))
            mix = torch.zeros_like(preprocess(img))
            #H, W, C
            for i in range(self.width):
                step = int(self.choice / self.width)
                rand = np.random.randint(1, (self.choice + 1) - step*i)
                mixed = _img[np.random.choice(np.arange(0, _img.shape[0]), rand, replace=False)]
                mixed = mixed.sum(axis=0)
                mixed = Image.fromarray(mixed.astype(np.uint8))
                mix += dirichlet[i] * preprocess(mixed)
            img = (beta * preprocess(img) + (1 - beta) * mix)
        else:
            img = preprocess(img)
        return img