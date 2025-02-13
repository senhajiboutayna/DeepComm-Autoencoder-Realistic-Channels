import torch
import numpy as np
import math

# To make plots about constellations
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import matplotlib.cm as cm

class MemoryMessages():
    """
    Small class to get samples at every epoch during training
    """
    def __init__(self, m):

        self.memory = np.arange(m)
        self.m = m

    def __len__(self):
        """
        To return the length remaining memory
        Returns:
            (int) memory size
        """
        return len(self.memory)
    
    def sample(self, batch_size=64):

        batch = []
        targets = []

        # Get batch_size samples
        for i in range(batch_size):
            # If we still have memory keep sampling
            if len(self.memory) > 0:
                # Get a random index from the memory
                idx = np.random.randint(0, len(self.memory))
                targets.append(self.memory[idx])
                batch.append(self.memory[idx])
                # Delete the sampled element from memory
                self.memory = np.delete(self.memory, idx)

            else:
                return np.array(batch), np.array(targets)
        
        return np.array(batch), np.array(targets)
