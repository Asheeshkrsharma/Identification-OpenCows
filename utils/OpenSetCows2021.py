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
        self, topDir, jsonPath, batchSize=0, split='train', eval=False, trackletChoiceProb = 0.5, maxSequenceLength=None, combine=False, transform=None, img_size=(224, 224)
    ):
        self.img_size = img_size
        self.maxSequenceLength = maxSequenceLength
        self.transform = True if transform != None else False
        self.topDir = topDir
        self.prob = trackletChoiceProb
        self.eval = eval
        self.batchSize = batchSize
        with open(jsonPath) as f:
            files = json.load(f)            
            self.dataset = files[split]
            self.lookup = {item['label']:[] for item in self.dataset}
            for i in range(len(self.dataset)):
                self.lookup[self.dataset[i]['label']].append(i)
            self.classWeights, self.classFrequency = self.__getClassWeights(files)
        self.numClasses = len(self.lookup)
        if transform != None:
            self.t = transform
            # self.t = transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # )

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
        return self.t(img)

    def __getClassWeights(self, dataSet):
        classFrequency = numpy.zeros(len(self.lookup))
        # for track in dataSet['train']:
        #     classFrequency[track['label']] += len(track['paths'])
        # for track in dataSet['test']:
        #     classFrequency[track['label']] += len(track['paths'])
        # weights = (1 / classFrequency / sum(classFrequency))
        # weights = weights / numpy.linalg.norm(weights)
        for track in dataSet['train']:
            classFrequency[track['label']] += 1
        for track in dataSet['test']:
            classFrequency[track['label']] += 1
        weights = sum(classFrequency) / (classFrequency * len(self.lookup))
        return weights, classFrequency

    # Get the number of items for this dataset (depending on the split)
    def __len__(self):
        if self.eval==True:
            return len(self.dataset)
        else:
            return max(len(self.lookup), self.batchSize)
    
    def loadImage(self, path):
        image = self.loadResizeImage(path)
        # Firstly, transform from NHWC -> NCWH
        image = image.transpose(2, 0, 1)
        # Now convert into pyTorch form
        image = torch.from_numpy(image).float() / 255
        if self.transform:
            image = self.t(image)
        return image
    
    def choose(self, choice, N, shuffle=False):
        # There can be a case where, Length of the sequence
        # is less than N. In that case, our strategy would
        # be to shuffle the sequence.
        L = len(choice)
        if L < N:
          X = random.randint(0, L)
          positive = [choice[i % L] for i in range(X, X+N)]
          random.shuffle(choice)
          anchor = [choice[i % L] for i in range(X, X+N)]
        else:
          S = len(choice) - N
          if S < N:
            X = random.randint(S, N)
          else:
            X = random.randint(N, S)
          anchor = [choice[i % L] if (i % L)<L else None for i in range(X, X+N)]
          positive = [choice[i % L] if (i % L)<L else None for i in range(X-N, X)]
        
        if shuffle == True:
            random.shuffle(positive)
        return anchor, positive

    def getSubset(self, category, index):
        candidates = self.lookup[category]
        candidates = list(set(candidates) - set([index]))
        candidates = random.choice(candidates) 
        return candidates

    def retrieve(self, paths):
        images = [os.path.join(self.topDir, image) for image in paths]
        return [self.loadImage(path) for path in images]

    # Index retrieval method
    def __getitem__(self, categories):
        # Sampler is going to return a tuple of category
        # We have to get a random index for each category
        # in the dataset
        try:
            catPos, catNeg = categories
            # Choose from indeces of a class in the dataset
            indexAnchor = random.choice(self.lookup[catPos]) 
            indexNeg = random.choice(self.lookup[catNeg])
        except TypeError as te:
            # Probably we are evaluating, in that case,
            # we just need to process one label
            if self.eval==False:
                # This means the dataloader return a sequence index
                # not a category.
                catPos = categories % len(self.lookup)
                indexAnchor = random.choice(self.lookup[catPos])
            else:
                catPos = self.dataset[categories]['label']
                indexAnchor = categories

            # And choose any random negative category
            catNeg = random.choice(list(set(range(len(self.lookup))) - set([catPos])))
            indexNeg = random.choice(self.lookup[catNeg])

        # If max sequence length is specified, we are training
        if self.maxSequenceLength != None:
            # By chance select a part of the same sequence to make
            # the positive and anchor samples.
            if random.random() >= self.prob:
                # Choose different sequences
                indexPos = self.getSubset(catPos, indexAnchor)
                _, positive = self.choose(
                    self.dataset[indexPos]['paths'],
                    self.maxSequenceLength
                )
                anchor, _ = self.choose(
                    self.dataset[indexAnchor]['paths'],
                    self.maxSequenceLength
                )
                assert indexPos != indexAnchor
            else:
                # Choose part of the same sequence
                sequence = self.dataset[indexAnchor]['paths']
                anchor, positive = self.choose(
                    sequence, self.maxSequenceLength, shuffle=True)

            # Next choose a negative sample
            sequence = self.dataset[indexNeg]['paths']
            negative, _ = self.choose(sequence, self.maxSequenceLength)

            anchor, positive, negative = (
                self.retrieve(anchor),
                self.retrieve(positive),
                self.retrieve(negative)
            )

            return torch.stack(negative), torch.stack(anchor), torch.stack(positive), catPos, catNeg
        else:
            # If max sequence length is not specified, we are evaluating
            sequence = self.dataset[indexAnchor]['paths']
            anchor = [os.path.join(self.topDir, image) for image in sequence]
            anchor = [self.loadImage(path) for path in anchor]
            return torch.stack(anchor), catPos

    # Index retrieval method
    # def __getitem__(self, index):
    #     # During evaluation, we need to consider all the tracklets in the dataset
    #     if self.eval==False:
    #         index = index % len(self.lookup)
    #         # Get the sequence using lookup
    #         indeces = self.lookup[index] # Returns indeces of a class in the dataset
    #         index = random.choice(indeces) # Choose a sequence from the list
    #     sequence, label = self.dataset[index]['paths'], self.dataset[index]['label']

    #     if self.maxSequenceLength != None:
    #       # Get a random yet temporally contgious set of images.
    #       anchor, positive = self.choose(sequence, self.maxSequenceLength)
          
    #       if random.random() < self.prob:
    #             try:
    #               positiveTemp = self.getSubset(label, index)
    #               # print(positiveTemp, index)
    #               positiveTemp = self.dataset[positiveTemp]['paths']
    #               _, positive = self.choose(positiveTemp, self.maxSequenceLength)
    #             except IndexError:
    #               pass

    #       # Get anchor
    #       anchor = [os.path.join(self.topDir, image) for image in anchor]
    #       anchor = [self.loadImage(path) for path in anchor]

    #       # Get a postive sequence
    #       positive = [os.path.join(self.topDir, image) for image in positive]
    #       positive = [self.loadImage(path) for path in positive]

    #       negative = None,
    #       negativeLabel = None
    #       # Should be quick enough
    #       while(True):
    #           exclusion = list(set(range(len(self.dataset))) - set([index]))
    #           exclusion = random.choice(exclusion)
    #           if self.dataset[exclusion]['label'] != label:
    #               sequence, negativeLabel = self.dataset[exclusion]['paths'], self.dataset[exclusion]['label']
    #               negative, _ = self.choose(sequence, self.maxSequenceLength)
    #               negative = [os.path.join(self.topDir, image) for image in negative]
    #               negative = [self.loadImage(path) for path in negative]
    #               break
    #       return torch.stack(negative), torch.stack(anchor), torch.stack(positive), label, negativeLabel
    #     else:
    #       anchor = [os.path.join(self.topDir, image) for image in sequence]
    #       anchor = [self.loadImage(path) for path in anchor]

    #       # It is assumed that a sequences has single class
    #       # label = numpy.asarray([1]) * int(label)
    #       # label = torch.from_numpy(label).long()
    #       return torch.stack(anchor), label
