import torch
import torch.nn.functional as F
import numpy as np
from math import erfc
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# To do block encoding (Hamming)
from sk_dsp_comm import fec_block as block

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


def bler(x, y):
    """
    Function to get the BLER
    Args:
        x (numpy array): Original samples
        y (numpy array): Decoded samples
    Returns:
        y of shape (batch_size, k): Decoded messages with Hamming
    """

    if len(x.shape) == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
    
    # Get the total number of messages
    batch_size, _ = x.shape  # Taille des Messages
    """
    Récupère le nombre de blocs dans x :
        batch_size : Nombre de messages dans le batch.
        k (ignoré avec _) : Longueur des messages.
    
    Exemple : Si x est de taille (10,4) (10 blocs, chacun de 4 bits), alors : 
        batch_size = 10
    """
    
    # Check where are the errors between received and transmitted 
    errors = np.not_equal(x, y)  ## Comparaison des Messages
    errors = np.atleast_2d(errors)
    """
    Résultat : Un tableau booléen où chaque élément est True si le bit est incorrect.
    """
    # How many errors per block
    errors_block = errors.sum(axis=1)  ## Nombre d'Erreurs par Bloc
    """
    axis=1 : La somme est effectuée sur chaque ligne.
    Exemple : 
    errors = [[False, False, True, False],
          [False, False, False, False]]

    errors_block = [1, 0]  # 1 erreur dans le 1er bloc, 0 dans le 2e.
    """
    # If there was an error in the block count it as bad block
    total_errors = (errors_block > 0).sum()   ## Comptage des Blocs Erronés 
    y = total_errors / batch_size

    return y

def plot_training_loss(losses):
    """
    Trace l'évolution de la perte pendant l'entraînement.
    """
    plt.figure(figsize=(8,5))
    plt.plot(losses, label="Perte d'entraînement")
    plt.xlabel("Itérations")
    plt.ylabel("Perte (Cross Entropy)")
    plt.title(f"Évolution de la perte pendant l'entraînement")
    plt.legend()
    plt.grid()
    plt.show()


def plot_constellation(encoder, m=16, h=None, title="Constellation", save_path=False):
    encoder.eval()
    with torch.no_grad():
        inputs = torch.arange(m).long().to(device)
        onehots = F.one_hot(inputs, num_classes=m).float()
        if h is not None:
            h_input = h.repeat(m, 1)
            x_encoded = encoder(onehots, h=h_input)
        else:
            x_encoded = encoder(inputs)

        # Représentation complexe (x = I + jQ)
        x_complex = x_encoded.view(m, -1, 2)  # reshape par symbole
        real = x_complex[..., 0].flatten()
        imag = x_complex[..., 1].flatten()

        plt.figure(figsize=(6,6))
        plt.scatter(real.cpu(), imag.cpu(), c='blue')
        for i, (x, y) in enumerate(zip(real, imag)):
            plt.text(x.cpu()+0.02, y.cpu(), str(i), fontsize=9)
        plt.grid(True)
        plt.title(title)
        plt.xlabel("In-phase")
        plt.ylabel("Quadrature")
        if save_path:
            plt.savefig(f"plots/{title}.png")
        plt.show()

# Générer une courbe théorique BPSK
def theoretical_ber(snr_db):
    snr = 10**(snr_db/10)
    return 0.5 * erfc(np.sqrt(snr))