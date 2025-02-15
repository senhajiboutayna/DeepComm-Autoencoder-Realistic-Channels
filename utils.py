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
    
def count_errors(inputs, targets):
    """
    Function to try count the errrors after Rx/Decoding wrt original messages (targets)
    Args:
        inputs pytorch tensor of shape(batch_size, m): 
        targets pytorch tensor of shape(batch_size): 
    Returns:
        total_errrors (float): Total errors found
    """
    # Each example i has m probabilities. Is the probabilit of example i being m message
    # Choose the highest probability
    chosen_input = torch.argmax(inputs, dim=1)
    
    # Get where both tensors are different
    errors = targets != chosen_input
    
    # Sum the errors to get the total
    total_errors = errors.sum().to("cpu").numpy()
    
    return total_errors
