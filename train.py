import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math

import matplotlib.pyplot as plt

from IPython.utils import io
from IPython import display

from channel import Channel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = True
