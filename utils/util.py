import os
import random

import numpy as np
import torch


def set_seed(seed, cuda=False):
    """
    Set the seed for the random number generators
    :param cuda:
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def create_directory(directory):
    """
    Create a directory if it does not exist
    :param directory:
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

