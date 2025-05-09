import torch
import numpy as np
import matplotlib.pyplot as plt

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
    errors = (x != y)  ## Comparaison des Messages
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

def plot_constellations(perfect_const, noisy_const, ml_const, modulation='qpsk', n_points=500):
    """
    Visualise et compare les constellations avec la constellation idéale.
    
    Args:
        perfect_const: Constellation avec CSI parfait
        noisy_const: Constellation avec feedback bruité sans ML
        ml_const: Constellation avec feedback bruité avec ML
        modulation: Type de modulation ('qpsk', '16qam', etc.)
        n_points: Nombre de points à afficher
    """
    # Définition des constellations idéales selon le type de modulation
    ideal_constellations = {
        'qpsk': np.array([
            [1/np.sqrt(2), 1/np.sqrt(2)],
            [-1/np.sqrt(2), 1/np.sqrt(2)],
            [-1/np.sqrt(2), -1/np.sqrt(2)],
            [1/np.sqrt(2), -1/np.sqrt(2)]
        ]),
        '16qam': np.array([
            [-3, -3], [-3, -1], [-3, 1], [-3, 3],
            [-1, -3], [-1, -1], [-1, 1], [-1, 3],
            [1, -3], [1, -1], [1, 1], [1, 3],
            [3, -3], [3, -1], [3, 1], [3, 3]
        ]) / np.sqrt(10)
    }
    
    ideal_points = ideal_constellations.get(modulation.lower(), None)
    if ideal_points is None:
        raise ValueError(f"Modulation {modulation} non supportée. Options: 'qpsk', '16qam'")
    
    # Sélection aléatoire de points pour la visualisation
    idx = np.random.choice(len(perfect_const), min(n_points, len(perfect_const)), replace=False)
    
    # Préparation des données
    perfect_points = np.vstack(perfect_const)[idx]
    noisy_points = np.vstack(noisy_const)[idx]
    ml_points = np.vstack(ml_const)[idx]
    
    # Création de la figure
    plt.figure(figsize=(18, 5))
    
    # 1. Constellation idéale
    plt.subplot(1, 4, 1)
    plt.scatter(ideal_points[:, 0], ideal_points[:, 1], color='k', marker='x', s=100, label='Points idéaux')
    plt.title(f'Constellation Idéale ({modulation.upper()})')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    # 2. CSI parfait
    plt.subplot(1, 4, 2)
    plt.scatter(ideal_points[:, 0], ideal_points[:, 1], color='k', marker='x', s=100, alpha=0.3)
    plt.scatter(perfect_points[:, 0], perfect_points[:, 1], alpha=0.6, color='b')
    plt.title('Constellation - CSI Parfait')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    # 3. Feedback bruité sans ML
    plt.subplot(1, 4, 3)
    plt.scatter(ideal_points[:, 0], ideal_points[:, 1], color='k', marker='x', s=100, alpha=0.3)
    plt.scatter(noisy_points[:, 0], noisy_points[:, 1], alpha=0.6, color='r')
    plt.title('Feedback Bruité (sans ML)')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    # 4. Feedback bruité avec ML
    plt.subplot(1, 4, 4)
    plt.scatter(ideal_points[:, 0], ideal_points[:, 1], color='k', marker='x', s=100, alpha=0.3)
    plt.scatter(ml_points[:, 0], ml_points[:, 1], alpha=0.6, color='g')
    plt.title('Feedback Bruité (avec ML)')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.show()