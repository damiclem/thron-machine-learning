# Dependencies
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class FashionDataset(Dataset):
    """ Fashion Dataset

    Multi-label dataset for classification of fashion images. The class allows to define the path where images are
    stored. Each image might have multiple labels associated. Labels are stored in the labels.csv file. Each label is
    associated to a unique number. Associations between numbers and labels can be found in the vocabulary.csv file.
    """

# Main
if __name__ == '__main__':
    # Not implemented
    raise NotImplementedError()