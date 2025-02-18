import torch
import matplotlib.pyplot as plt 
import numpy as np
import math
from scipy import special
from scipy.stats import norm, rayleigh, rice

def channel(x, n, k, snr_db, chann_type, K_rician):
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

        print(f"Mean of noise - AWGN: {torch.mean(var_channel * torch.randn_like(x))}, Std of noise - AWGN: {torch.std(var_channel * torch.randn_like(x))}")

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

        print(f"Mean x_channel - Rayleigh: {torch.mean(x_channel)}, Std x_channel - Rayleigh: {torch.std(x_channel)}")

    elif chann_type == "Rician":
        K_rician = torch.tensor(K_rician, dtype=torch.float32)  # Convertir en tenseur
        fading = torch.sqrt(K_rician / (K_rician + 1)) + torch.sqrt(1 / (K_rician + 1)) * torch.abs(torch.randn_like(x) + 1j * torch.randn_like(x))
        x_channel = fading.real * x + var_channel * torch.randn_like(x)

        print(f"Mean x_channel - Rician: {torch.mean(x_channel)}, Std x_channel - Rician: {torch.std(x_channel)}")

    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}")

    return x_channel


def plot_channel_distribution_AWGN(snr_db, n_samples):
    """
    Vérifie la distribution du signal après passage dans un canal donné.
    """
    n, k = 7, 4  # Exemple de dimensions
    x = torch.randn(n_samples, n)  # Signal d'entrée simulé

    # Appliquer le canal
    x_channel = channel(x, n, k, snr_db, chann_type = "AWGN", K_rician=3)

    # Tracer l'histogramme
    plt.figure(figsize=(8, 5))

    x_channel_np = x_channel.numpy().flatten()  # Convertir en numpy

    plt.hist(x_channel_np, bins=50, density=True, alpha=0.6, label="Simulated")
    
    x_range = np.linspace(x_channel_np.min(), x_channel_np.max(), 100)  
    plt.plot(x_range, norm.pdf(x_range, 0, 1/np.sqrt(2)), 'r-', label='Theoretical')
    plt.xlabel("Valeur du signal")
    plt.ylabel("Densité")
    plt.title(f"Distribution du signal après le canal AWGN")
    plt.legend()
    plt.grid()
    plt.show()

def plot_channel_distribution_Rayleigh(snr_db=10, n_samples=10000):
    """
    Vérifie la distribution du signal après passage dans un canal donné.
    """
    n, k = 7, 4  # Exemple de dimensions
    x = torch.randn(n_samples, n)  # Signal d'entrée simulé

    # Appliquer le canal
    x_channel = channel(x, n, k, snr_db, chann_type = "Rayleigh", K_rician = 3)

    x_channel_np = x_channel.numpy().flatten()  # Convertir en numpy

    # Tracer l'histogramme
    plt.figure(figsize=(8, 5))
    plt.hist(x_channel_np, bins=50, density=True, alpha=0.6, label=f"Simulated")

    x_range = np.linspace(0, 8, 500)
    plt.plot(x_range, rayleigh.pdf(x_range, scale=1 / np.sqrt(2)), 'r-', label='Theoretical')

    plt.xlabel("Valeur du signal")
    plt.ylabel("Densité")
    plt.title(f"Distribution du signal après le canal Rayleigh")
    plt.legend()
    plt.grid()
    plt.show()

def plot_channel_distribution_Rician(snr_db=10, n_samples=10000, K_rician=3):
    """
    Vérifie la distribution du signal après passage dans un canal donné.
    """
    n, k = 7, 4  # Exemple de dimensions
    x = torch.randn(n_samples, n)  # Signal d'entrée simulé

    # Appliquer le canal
    x_channel = channel(x, n, k, snr_db, chann_type = "Rician", K_rician = K_rician)

    x_channel_np = x_channel.numpy().flatten()  # Convertir en numpy

    # Tracer l'histogramme
    plt.figure(figsize=(8, 5))
    plt.hist(x_channel.numpy().flatten(), bins=50, density=True, alpha=0.6, label=f"Simulated")
    
    x_range = np.linspace(0, 8, 500)
    plt.plot(x_range, rice.pdf(x_range, K_rician, scale=1 / np.sqrt(2)), 'r-', label='Theoretical')

    plt.xlabel("Valeur du signal")
    plt.ylabel("Densité")
    plt.title(f"Distribution du signal après le canal Rician")
    plt.legend()
    plt.grid()
    plt.show()

def plot_fading_distributions():
    """Affichage des distributions Rayleigh et Rician pour différentes valeurs de K."""
    plt.figure(figsize=(8, 5))

    K_values = [0, 1, 2, 4]  # Valeurs de K testées
    x_range = np.linspace(0, 8, 500)

    for K in K_values:
        if K == 0:
            pdf = rayleigh.pdf(x_range, scale=1 / np.sqrt(2))
            label = "K=0 (Rayleigh)"
        else:
            pdf = rice.pdf(x_range, K, scale=1 / np.sqrt(2))
            label = f"K={K}"

        plt.plot(x_range, pdf, label=label)

    plt.xlabel("Received Signal Envelope Voltage r (volts)")
    plt.ylabel("P(r)")
    plt.title("Rayleigh et Rician Fading Distributions")
    plt.legend()
    plt.grid()
    plt.show()

plot_channel_distribution_AWGN(snr_db=10, n_samples=10000)
plot_channel_distribution_Rayleigh(snr_db=10, n_samples=10000)
plot_channel_distribution_Rician(snr_db=10, n_samples=10000, K_rician=3)

# Tracé des différentes distributions de fading
plot_fading_distributions()