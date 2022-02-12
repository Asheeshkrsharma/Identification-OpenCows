import os, time, json
import numpy

import torch
from torch.utils import data
from torchvision import transforms

from PIL import Image

import random

class OpenSetCows2021(data.Dataset):
    # Class constructor
    def __init__(
        self, topDir, jsonPath, combine=False, transform=False, img_size=(224, 224)
    ):
        self.img_size = img_size
        self.transform = transform
        dataSet = {"test": [], "train": [], "valid": []}
        self.flatFiles = []
        with open(jsonPath) as f:
            files = json.load(f)
            # Prep the files
            for parentDir in files.keys():
                for subDir in files[parentDir].keys():
                    for split in files[parentDir][subDir]:
                        timestamps = []
                        paths = []
                        for image in files[parentDir][subDir][split]:
                            path = os.path.join(topDir, parentDir, subDir, image)
                            paths.append(path)
                            timestamps.append(os.path.getctime(path))
                        indeces = numpy.argsort(timestamps)
                        if len(paths) > 0:
                            dataSet[split].append({"paths": paths, "sort": indeces})
                            self.flatFiles += paths
        
        self.dataSet = dataSet
        self.t = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # Load an image into memory, pad it to img size with a black background
    def loadResizeImage(self, img_path):
        size = self.img_size
        # Load the image
        img = Image.open(img_path)

        # Keep the original image size
        old_size = img.size

        # Compute resizing ratio
        ratio = float(size[0]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # Actually resize it
        img = img.resize(new_size, Image.ANTIALIAS)

        # Paste into centre of black padded image
        new_img = Image.new("RGB", (size[0], size[1]))
        new_img.paste(img, ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2))

        # Convert to numpy
        new_img = numpy.array(new_img, dtype=numpy.uint8)

        return new_img

    # Transform the numpy images into pyTorch form
    def transformImages(self, img):
        # Firstly, transform from NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        # Now convert into pyTorch form
        img = torch.from_numpy(img).float() / 255
        return self.t(img)

    # Get the number of items for this dataset (depending on the split)
    def __len__(self):
        return len(self.flatFiles)

    # Index retrieval method
    def __getitem__(self, index):
        pth = self.flatFiles[index]
        image = self.loadResizeImage(pth)
        if self.transform:
            image = self.transformImages(image)
        return image, pth

class OpenSetCows2021TrackLet(data.Dataset):
    # Class constructor
    def __init__(
        self, topDir, jsonPath, maxSequenceLength=5, combine=False, transform=False, img_size=(224, 224)
    ):
        self.img_size = img_size
        self.maxSequenceLength = maxSequenceLength
        self.transform = transform
        self.topDir = topDir
        with open(jsonPath) as f:
            files = json.load(f)
            self.dataset = files['train']
        self.t = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # Load an image into memory, pad it to img size with a black background
    def loadResizeImage(self, img_path):
        size = self.img_size
        # Load the image
        img = Image.open(img_path)

        # Keep the original image size
        old_size = img.size

        # Compute resizing ratio
        ratio = float(size[0]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # Actually resize it
        img = img.resize(new_size, Image.ANTIALIAS)

        # Paste into centre of black padded image
        new_img = Image.new("RGB", (size[0], size[1]))
        new_img.paste(img, ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2))

        # Convert to numpy
        new_img = numpy.array(new_img, dtype=numpy.uint8)

        return new_img

    # Transform the numpy images into pyTorch form
    def transformImages(self, img):
        # Firstly, transform from NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        # Now convert into pyTorch form
        img = torch.from_numpy(img).float() / 255
        return self.t(img)

    # Get the number of items for this dataset (depending on the split)
    def __len__(self):
        return len(self.dataset)
    
    def loadImage(self, path):
        image = self.loadResizeImage(path)
        if self.transform:
            image = self.transformImages(image)
        return image
    
    def choose(self, choice, N):
        X = random.randint(0, len(choice) - N)
        L = len(choice)
        return [choice[i % L] if (i % L)<L else None for i in range(X, X+N)]

    # Index retrieval method
    def __getitem__(self, index):
        sequence, label = self.dataset[index]['paths'], self.dataset[index]['label']
        # Get a random yet temporally contgious set of images.
        sequence = self.choose(sequence, self.maxSequenceLength)
        sequence = [os.path.join(self.topDir, image) for image in sequence]
        images = [self.loadImage(path) for path in sequence]

        # It is assumed that a sequences has single class
        # label = numpy.asarray([1]) * int(label)
        # label = torch.from_numpy(label).long()

        return torch.stack(images), label