import torch
import numpy as np
import math

def channel(x, n, k, snr_db, chann_type="AWGN"):
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
    # Convertir x en tenseur PyTorch s'il est sous forme NumPy
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    
    if chann_type == "AWGN":
        # Finally calculate the variance of the AWGN
        n0 = 1/(snr_lin*rate)
        var_channel = math.sqrt(n0/2)

        # Use the reparametrization trick to apply noise to x
        if torch.is_tensor(x):
            x_channel = x + var_channel*torch.randn_like(x)  # Génère un bruit gaussien de même forme que x
        else:
            x_channel = x + var_channel*np.random.randn(*x.shape)  # Génère un bruit gaussien pour x.

    return x_channel