# Dependencies
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import json
import os


class FashionDataset(Dataset):
    """ Fashion Dataset

    Multi-label dataset for classification of fashion images. The class allows to define the path where images are
    stored. Each image might have multiple labels associated. Labels are stored in the labels.csv file. Each label is
    associated to a unique number. Associations between numbers and labels can be found in the vocabulary.csv file.
    """

    # Constructor
    def __init__(self, images_dir, tokens_json, vocabulary_json, transform=None):
        """
        Constructor

        :param images_dir:  Path to images directory
        :param tokens_json: Path to file containing images tags
        :param vocabulary_json: Path to file containing vocabulary
        :param transform: Transformation to be executed on images
        """
        # Store path to images directory
        self.images_dir = images_dir
        # Open tokens JSON file
        with open(tokens_json, 'r') as tokens_file:
            # Load object mapping image to tokens
            self.tokens = json.load(tokens_file)
            # Initialize index (keep only if image exists)
            self.index = []
            # Loop through each key in given tokens
            for key in self.tokens.keys():
                # Define path to image
                image_path = os.path.join(self.images_dir, '%s.jpg' % key)
                # Case associated image does not exist
                if not os.path.isfile(image_path):
                    # Then, skip iteration
                    continue
                # Otherwise, update index
                self.index.append(key)
        # Open vocabulary JSON file
        with open(vocabulary_json, 'r') as vocabulary_file:
            # Load dataframe mapping token to index
            self.vocabulary = json.load(vocabulary_file)
        # Store transformation
        self.transform = transform

    # Length
    def __len__(self):
        return len(self.index)

    # Brackets
    def __getitem__(self, i):
        # Define index
        index = self.index[i]
        # Load image from folder
        image = Image.open(os.path.join(self.images_dir, '%s.jpg' % index))
        # Cast image to numpy array (and normalize between 0,1
        image = np.asarray(image.convert('RGB'), dtype=int) / 255
        # # Debug
        # if image.shape != (80, 60, 3):
        #     raise Exception('Shape %s does not match expected' % str(image.shape))
        # Define tokens list and vocabulary
        tokens, vocabulary = self.tokens[index], self.vocabulary
        # Initialize tokens vector
        encoded = np.array([0] * len(self.vocabulary), dtype=int)
        # Update tokens vector
        encoded[tokens] = 1
        # Initialize sample
        sample = {'image': image, 'tokens': encoded}
        # Eventually, apply transformation to sample
        sample = self.transform(sample) if self.transform else sample
        # Return sample
        return sample


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors. """

    def __call__(self, sample):
        # Unpack image and tokens
        image, tokens = sample['image'], sample['tokens']
        # Swap color axis (H x W x C -> C x H x W)
        image = image.transpose((2, 0, 1))
        #  Pass results on
        return {'image': torch.from_numpy(image).type(torch.float), 'tokens': torch.from_numpy(tokens)}


class Resize(object):
    """ Resizes images in sample to target size. """

    def __init__(self, size):
        # Call parent constructor
        super().__init__()
        # Store target shape
        self.size = size
        # Store target transformation
        self.resize = transforms.Resize(size=size)

    def __call__(self, sample):
        # Retrieve image and tokens from sample
        image, tokens = sample['image'], sample['tokens']
        # Resize image
        image = self.resize(image)
        # Return updated sample
        return {'image': image, 'tokens': tokens}


def train_test_split(data, p=0.8):
    raise NotImplementedError


# Main
if __name__ == '__main__':

    # Not implemented
    raise NotImplementedError
