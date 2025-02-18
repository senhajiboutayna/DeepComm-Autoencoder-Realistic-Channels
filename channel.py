import torch
import matplotlib.pyplot as plt 
import numpy as np
import math

def channel(x, n, k, snr_db, chann_type="AWGN", K_rician=5):
    """
    Definition of the channel. e.g. Add AWGN to the samples
    Args:
        x of shape (batch_size, k): Encoded messages
        n (int): Length of the encoded messages
        k (int): Length of the actual messages
        snr_db (float): SNR to add noise
        chann_type (string): Channel type. Currently only AWGN available
    Returns:
        x_channel of shape (batch_size, k): x with noise
    """
    # Conversion du SNR dB en Valeur Linéaire
    """
    Le SNR est souvent exprimé en décibels (dB), une échelle logarithmique. Cette ligne convertit le SNR en une valeur linéaire.
    """
    snr_lin = 10**(snr_db/10)
    
    # Get the rate of the encoding
    rate = k/n   # Rapport entre la longueur des messages originaux (k) et celle des messages encodés (n).
    n0 = 1 / (snr_lin * rate)  # Noise variance
    # Finally calculate the variance of the AWGN
    var_channel = math.sqrt(n0/2)
    # Convertir x en tenseur PyTorch s'il est sous forme NumPy
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    
    if chann_type == "AWGN":

        # Use the reparametrization trick to apply noise to x
        if torch.is_tensor(x):
            x_channel = x + var_channel*torch.randn_like(x)  # Génère un bruit gaussien de même forme que x
        else:
            x_channel = x + var_channel*np.random.randn(*x.shape)  # Génère un bruit gaussien pour x.

    elif chann_type == "Rayleigh":
        # Rayleigh fading channel
        fading = torch.abs(torch.randn_like(x) + 1j * torch.randn_like(x)) / math.sqrt(2)
        n0 = 1 / (snr_lin * rate)
        var_channel = math.sqrt(n0 / 2)
        if torch.is_tensor(x):
            noise = var_channel * torch.randn_like(x)
            x_channel = fading * x + noise
        else:
            fading = np.abs(np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)) / math.sqrt(2)
            noise = var_channel * np.random.randn(*x.shape)
            x_channel = fading * x + noise

    elif chann_type == "Rician":
        K_rician = torch.tensor(K_rician, dtype=torch.float32)  # Convertir en tenseur
        fading = torch.sqrt(K_rician / (K_rician + 1)) + torch.sqrt(1 / (K_rician + 1)) * torch.abs(torch.randn_like(x) + 1j * torch.randn_like(x))
        x_channel = fading.real * x + var_channel * torch.randn_like(x)

    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}")

    return x_channel


def plot_channel_distribution(chann_type, snr_db=10, n_samples=10, K_rician=3):
    """
    Vérifie la distribution du signal après passage dans un canal donné.
    """
    n, k = 7, 4  # Exemple de dimensions
    x = torch.randn(n_samples, n)  # Signal d'entrée simulé

    # Appliquer le canal
    x_channel = channel(x, n, k, snr_db, chann_type, K_rician)

    # Tracer l'histogramme
    plt.figure(figsize=(8, 5))
    plt.hist(x_channel.numpy().flatten(), bins=50, density=True, alpha=0.6, label=f"{chann_type} Channel")
    plt.xlabel("Valeur du signal")
    plt.ylabel("Densité")
    plt.title(f"Distribution du signal après le canal {chann_type}")
    plt.legend()
    plt.grid()
    plt.show()

# Tester les trois canaux
plot_channel_distribution("AWGN")
plot_channel_distribution("Rayleigh")
plot_channel_distribution("Rician")