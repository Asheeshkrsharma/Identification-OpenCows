# Core libraries
import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import matplotlib.patches as mpatches
from torchvision import transforms

# PyTorch
import torch
from torch.utils import data

# Local libraries
from PIL import Image

"""
File contains input/output utility functions
"""

# Create a sorted list of all files with a given extension at a given directory
# If full_path is true, it will return the complete path to that file
def allFilesAtDirWithExt(directory, file_extension, full_path=True):
    # Make sure we're looking at a folder
    if not os.path.isdir(directory): print(directory)
    assert os.path.isdir(directory)

    # Gather the files inside
    if full_path:
        files = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]
    else:
        files = [x for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]

    return files

# Similarly, create a sorted list of all folders at a given directory
def allFoldersAtDir(directory, full_path=True):
    # Make sure we're looking at a folder
    if not os.path.isdir(directory): print(directory)
    assert os.path.isdir(directory)

    # Find all the folders
    if full_path:
        folders = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]
    else:
        folders = [x for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]

    return folders

# Load an image into memory, pad it to img size with a black background
def loadResizeImage(img_path, size):      
    # Load the image
    img = Image.open(img_path)

    # Keep the original image size
    old_size = img.size

    # Compute resizing ratio
    ratio = float(size[0])/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # Actually resize it
    img = img.resize(new_size, Image.ANTIALIAS)

    # Paste into centre of black padded image
    new_img = Image.new("RGB", (size[0], size[1]))
    new_img.paste(img, ((size[0]-new_size[0])//2, (size[1]-new_size[1])//2))

    # Convert to numpy
    new_img = np.array(new_img, dtype=np.uint8)

    return new_img


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


"""
Manages loading the dataset into a PyTorch form
"""


class OpenSetCows2020(data.Dataset):
    # Class constructor
    def __init__(
        self,
        fold,
        fold_file,
        split="train",
        combine=False,
        known=True,
        transform=False,
        img_size=(224, 224),
        suppress_info=True,
    ):
        """
        Class attributes
        """

        # The root directory for the dataset itself
        self.__root = "datasets/OpenSetCows2020"

        # The fold we're currently considering
        self.__fold = str(fold)

        # The file containing the category splits for this fold
        self.__fold_file = fold_file

        # The split we're after (e.g. train/test)
        self.__split = split

        # Whether we should just load everything
        self.__combine = combine

        # Whether we're after known or unknown categories, irrelevant if combine is true
        self.__known = known

        # Whether to transform images/labels into pyTorch form
        self.__transform = transform

        # The directory containing actual imagery
        self.__train_images_dir = os.path.join(self.__root, "images/train")
        self.__test_images_dir = os.path.join(self.__root, "images/test")

        # Retrieve the number of classes from these
        self.__train_folders = allFoldersAtDir(self.__train_images_dir)
        self.__test_folders = allFoldersAtDir(self.__test_images_dir)
        assert len(self.__train_folders) == len(self.__test_folders)
        self.__num_classes = len(self.__train_folders)

        # Load the folds dictionary containing known and unknown categories for each fold
        if os.path.exists(self.__fold_file):
            with open(self.__fold_file, "rb") as handle:
                self.__folds_dict = json.load(handle)
        else:
            print(f"File path doesn't exist: {self.__fold_file}")
            sys.exit(1)

        # A quick check
        assert self.__fold in self.__folds_dict.keys()

        # The image size to resize to
        self.__img_size = img_size

        # A dictionary storing seperately the list of image filepaths per category for
        # training and testing
        self.__sorted_files = {}

        # A dictionary storing separately the complete lists of filepaths for training and
        # testing
        self.__files = {}
        """
		Class setup
		"""

        # Create dictionaries of categories: filepaths
        train_files = {
            os.path.basename(f): allFilesAtDirWithExt(f, ".jpg")
            for f in self.__train_folders
        }
        test_files = {
            os.path.basename(f): allFilesAtDirWithExt(f, ".jpg")
            for f in self.__test_folders
        }

        # List of categories to be removed
        remove = []

        # Should we be just returning all the classes (not removing unknown or known classes)
        if not self.__combine:
            # Create a list of categories to be removed according to whether we're after known
            # or unknown classes
            if self.__known:
                remove = self.__folds_dict[self.__fold]["unknown"]
            else:
                remove = self.__folds_dict[self.__fold]["known"]

        # Remove these from the dictionaries (might not remove anything)
        self.__sorted_files["train"] = {
            k: v for (k, v) in train_files.items() if k not in remove
        }
        self.__sorted_files["test"] = {
            k: v for (k, v) in test_files.items() if k not in remove
        }

        # Consolidate this into one long list of filepaths for training and testing
        train_list = [v for k, v in self.__sorted_files["train"].items()]
        test_list = [v for k, v in self.__sorted_files["test"].items()]
        self.__files["train"] = [item for sublist in train_list for item in sublist]
        self.__files["test"] = [item for sublist in test_list for item in sublist]

        # Report some things
        if not suppress_info:
            self.printStats()

    """
	Superclass overriding methods
	"""

    # Get the number of items for this dataset (depending on the split)
    def __len__(self):
        return len(self.__files[self.__split])

    # Index retrieval method
    def __getitem__(self, index):
        # Get and load the anchor image
        img_path = self.__files[self.__split][index]

        # Load the anchor image
        img_anchor = loadResizeImage(img_path, self.__img_size)

        # Retrieve the class/label this index refers to
        current_category = self.__retrieveCategoryForFilepath(img_path)

        # Get a positive (another random image from this class)
        img_pos = self.__retrievePositive(current_category, img_path)

        # Get a negative (a random image from a different random class)
        img_neg, label_neg = self.__retrieveNegative(current_category, img_path)

        # Convert all labels into numpy form
        label_anchor = np.array([int(current_category)])
        label_neg = np.array([int(label_neg)])

        # For sanity checking, visualise the triplet
        # self.__visualiseTriplet(img_anchor, img_pos, img_neg, label_anchor)

        # Transform to pyTorch form
        if self.__transform:
            img_anchor, img_pos, img_neg = self.__transformImages(
                img_anchor, img_pos, img_neg
            )
            label_anchor, label_neg = self.__transformLabels(label_anchor, label_neg)

        label_anchor, label_neg = label_anchor.view(len(label_anchor)), label_neg.view(len(label_neg))
        return img_anchor, img_pos, img_neg, label_anchor, label_neg

    """
	Public methods
	"""

    # Print stats about the current state of this dataset
    def printStats(self):
        knowns = len(self.__folds_dict[self.__fold]["known"])
        unknowns = len(self.__folds_dict[self.__fold]["unknown"])
        combinations = "on" if self.__combine else "off"
        print(
            f"This dataset ({self.__split} split, {int(self.__fold)+1} fold) has {knowns} known and {unknowns} unknown categories, with combinations turned {combinations}."
        )
        print(
            f"It has {len(self.__files['train'])} images in the train set and {len(self.__files['test'])} in the test set."
        )
        print(
            "You can try to plot the triplet samples by `dataset.plotSamples(<number of samples>)`, and see the distribution by `dataset.distribution()`"
        )
        # print("Loaded the OpenSetCows2019 dataset_____________________________")
        # print(f"Fold = {int(self.__fold)+1}, split = {self.__split}, combine = {self.__combine}, known = {self.__known}")
        # print(f"Found {self.__num_classes} categories: {len(self.__folds_dict[self.__fold]['known'])} known, {len(self.__folds_dict[self.__fold]['unknown'])} unknown")
        # print(f"With {len(self.__files['train'])} train images, {len(self.__files['test'])} test images")
        # print(f"Unknown categories {self.__folds_dict[self.__fold]['unknown']}")
        # print("_______________________________________________________________")

    """
	(Effectively) private methods
	"""

    # Created by Asheesh Sharma
    def distribution(self):
        dt = self.getDistribution()
        unknown = self.getUnknown()
        cat = []
        nums = []
        c = []
        for category, filepaths in dt:
            cat.append('$C_{'+str(int(category))+'}$')
            nums.append(len(filepaths))
            if category in unknown:
                c.append("#de4a29")
            else:
                c.append("#a278aa")
        plt.bar(cat, nums, color=c,width = 0.8)
        plt.xticks(cat, cat)
        plt.tick_params(rotation=-45)
        red_patch = mpatches.Patch(color="#de4a29", label="Unknown")
        blue_patch = mpatches.Patch(color="#a278aa", label="Known")
        plt.title("Identification Instance Distribution")
        plt.xlabel("Class (Individual Animals)")
        plt.ylabel("Instances")
        plt.legend(handles=[red_patch, blue_patch])
        # Titles and labels
        plt.tight_layout(pad=0.9)
        
        plt.xlim(-0.4,45.4)

    def getUnknown(self):
        return self.__folds_dict[self.__fold]["unknown"]

    # Created by Asheesh Sharma
    def plotSamples(self, num):
        for i in range(num):
            classID = randint(0, self.getNumClasses())
            anchor, positive, negative, anchorLabel, negativeLabel = self.__getitem__(
                classID
            )
            unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            plt.imshow(unorm(negative).permute(1, 2, 0))
            ax.set_title("Negative")
            plt.axis("off")
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow(unorm(anchor).permute(1, 2, 0))
            ax.set_title("Anchor")
            plt.axis("off")
            ax = fig.add_subplot(1, 3, 3)
            plt.imshow(unorm(positive).permute(1, 2, 0))
            ax.set_title("Positive")
            plt.axis("off")
        plt.show()

    # Transform the numpy images into pyTorch form
    def __transformImages(self, img_anchor, img_pos, img_neg):
        t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Firstly, transform from NHWC -> NCWH
        img_anchor = img_anchor.transpose(2, 0, 1)
        img_pos = img_pos.transpose(2, 0, 1)
        img_neg = img_neg.transpose(2, 0, 1)

        # Now convert into pyTorch form
        img_anchor = torch.from_numpy(img_anchor).float() / 255
        img_pos = torch.from_numpy(img_pos).float() / 255
        img_neg = torch.from_numpy(img_neg).float() / 255

        img_anchor, img_pos, img_neg = t(img_anchor), t(img_pos), t(img_neg)
        return img_anchor, img_pos, img_neg

    # Transform the numpy labels into pyTorch form
    def __transformLabels(self, label_anchor, label_neg):
        # Convert into pyTorch form
        label_anchor = torch.from_numpy(label_anchor).long()
        label_neg = torch.from_numpy(label_neg).long()

        return label_anchor, label_neg

    # Print some info about the distribution of images per category
    def __printImageDistribution(self):
        for category, filepaths in self.__sorted_files[self.__split].items():
            print(category, len(filepaths))

    # For a given filepath, return the category which contains this filepath
    def __retrieveCategoryForFilepath(self, filepath):
        # Iterate over each category
        for category, filepaths in self.__sorted_files[self.__split].items():
            if filepath in filepaths:
                return category

    # Get another positive sample from this class
    def __retrievePositive(self, category, filepath):
        # Copy the list of possible positives and remove the anchor
        possible_list = list(self.__sorted_files[self.__split][category])
        assert filepath in possible_list
        possible_list.remove(filepath)

        # Randomly select a filepath
        img_path = random.choice(possible_list)

        # Load and return the image
        img = loadResizeImage(img_path, self.__img_size)
        return img

    def __retrieveNegative(self, category, filepath):
        # Get the list of categories and remove that of the anchor
        possible_categories = list(self.__sorted_files[self.__split].keys())
        assert category in possible_categories
        possible_categories.remove(category)

        # Randomly select a category
        random_category = random.choice(possible_categories)

        # Randomly select a filepath in that category
        img_path = random.choice(self.__sorted_files[self.__split][random_category])

        # Load and return the image along with the selected label
        img = loadResizeImage(img_path, self.__img_size)
        return img, random_category

    """
	Getters
	"""
    # Print some info about the distribution of images per category
    def getDistribution(self):
        return self.__sorted_files[self.__split].items()

    def getNumClasses(self):
        return self.__num_classes

    def getNumTrainingFiles(self):
        return len(self.__files["train"])

    def getNumTestingFiles(self):
        return len(self.__files["test"])

    """
	Setters
	"""

    """
	Static methods
	"""


# Entry method/unit testing method
if __name__ == "__main__":
    pass
