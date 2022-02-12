import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

# Usual math
import numpy

from sklearn.neighbors import KNeighborsClassifier

from utils.OpenSetCows2020 import OpenSetCows2020


def experiment(modelClass, foldFile, lRate=1e-5, decay=1e-4, momentum=0.9):
    """Setup an experiment for training the Resnet network

    Parameters
    ----------
    modelClass : nn.Module class Pytorch model constructor.
        nn.Module
    foldFile : JSON file path containing train/test split, from the Opencows2020 dataset
        Found in ./OpenSetCows2020/indentification/splits
    lRate : Learning rate, optional
        Defaults to 1e-5
    decay : Weight decay, optional
        Defaults to 1e-4
    momentum : Stochastic Gradient Descent optimiser variable, optional
        Defaults to 0.9

    Returns
    -------
    device: Checks devices, returns CUDA if GPU available
        str
    model: An instance of the Resnet model
        nn.Module
    test: testing dataset
        nn.DataLoader
    train: raining dataset
        nn.DataLoader
    optimiser: Stochastic Gradient Descent optimiser
        nn.Optimizer
    """
    # Get the device, prefer Cuda over CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup test and train data loaders
    # Arguments taken as specified in the Paper.
    testDataset = OpenSetCows2020(
        0,
        foldFile,
        split="test",
        transform=True,
        combine=True,
        suppress_info=True,
    )
    trainDataset = OpenSetCows2020(
        0,
        foldFile,
        split="train",
        transform=True,
        combine=True,
        suppress_info=True,
    )

    testDataLoader = DataLoader(
        testDataset,
        batch_size=16,
        num_workers=2,
        shuffle=True,
    )
    trainDataLoader = DataLoader(
        trainDataset,
        batch_size=16,
        num_workers=2,
        shuffle=True,
    )

    # Model
    model = modelClass(trainDataset.getNumClasses())
    model.to(device)
    optimiser = optim.SGD(
        model.parameters(), momentum=momentum, lr=lRate, weight_decay=decay
    )
    return device, model, testDataLoader, trainDataLoader, optimiser


def evaluate(model, dataLoader, device, lossFn):
    """Evaluates a loss on a dataset

    Parameters
    ----------
    model : Resnet model
        nn.Module
    dataLoader : testing/training dataset
        nn.DataLoader
    device : a GPU or a CPU
        str
    lossFn : Loss function
        nn.Module

    Returns
    -------
    Loss: Evaluated loss
        Number
    """
    model.eval()
    losses = []
    for step, (anchors, positives, negatives, anchorLabel, negativeLabel) in enumerate(
        dataLoader
    ):
        anchors, positives, negatives, anchorLabel, negativeLabel = (
            anchors.to(device),
            positives.to(device),
            negatives.to(device),
            anchorLabel.view(len(anchorLabel)).to(device),
            negativeLabel.view(len(negativeLabel)).to(device),
        )
        # The forward method returns three embeddings
        negativeEMBD, anchorEMBD, positiveEMBD, prediction = model(
            negatives, anchors, positives
        )
        loss = lossFn(
            negativeEMBD,
            anchorEMBD,
            positiveEMBD,
            prediction,
            torch.cat((anchorLabel, anchorLabel, negativeLabel), dim=0),
        )
        losses.append(loss.data)
        break
    return sum(losses) / len(losses)


def inferAll(model, dataLoader, device, Num=200):
    """Infers a select number of images and returns the embeddings

    Parameters
    ----------
    model : Resnet model
        nn.Module
    dataLoader : testing/training dataset
        nn.DataLoader
    device : a GPU or a CPU
        str
    Num : Number of images to infer
        Number

    Returns
    -------
    Embeddings: 128-dimensional model output
        List of torch tensors: [Num, 128]
    Labels: Labels corresponding to the embeddings
        List
    """

    outputEMBDs, labelEMBDS = numpy.zeros((1, 128)), numpy.zeros((1))
    breakStep = 0
    model.eval()

    if Num > 6000:
      for images, _, _, labels, _ in tqdm(dataLoader):
          embeddings, _ = model(images.to(device))
          # Convert the data to numpy format
          embeddings, labels = (
              embeddings.data.cpu().numpy(),
              labels.view(len(labels)).cpu().numpy(),
          )
          # Store testing data on this batch ready to be evaluated
          outputEMBDs, labelEMBDS = numpy.concatenate(
              (outputEMBDs, embeddings), axis=0
          ), numpy.concatenate((labelEMBDS, labels), axis=0)
    else:
      for step, (images, _, _, labels, _) in enumerate(dataLoader):
          embeddings, _ = model(images.to(device))
          # Convert the data to numpy format
          embeddings, labels = (
              embeddings.data.cpu().numpy(),
              labels.view(len(labels)).cpu().numpy(),
          )
          # Store testing data on this batch ready to be evaluated
          outputEMBDs, labelEMBDS = numpy.concatenate(
              (outputEMBDs, embeddings), axis=0
          ), numpy.concatenate((labelEMBDS, labels), axis=0)
          breakStep += len(labels)
          if breakStep > Num:
              break
    return outputEMBDs, labelEMBDS


# Function to compose model input images in a grid
# compose = lambda images, horizontal: numpy.concatenate(
#     [img.permute(1, 2, 0) / 255 if horizontal == 1 else img for img in images],
#     axis=horizontal,
# )

# composite = compose(
#         [compose(negatives.cpu()[:3,], 1), compose(anchors.cpu()[:3,], 1), compose(positives.cpu()[:3,], 1)], 0
# )