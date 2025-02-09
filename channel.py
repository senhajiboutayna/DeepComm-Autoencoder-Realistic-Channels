import torch
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


class Channel:

    def __init__(self,k,n,x, snr_db, chann_type):
        self.snr_db = snr_db
        self.chann_type = chann_type
        self.k = k
        self.n = n
        self.x = x

        snr_lin = 10 ** (snr_db / 10)   # Conversion dB -> linéaire
        # Get the rate of the encoding
        rate = k/n   # Rapport entre la longueur des messages originaux (k) et celle des messages encodés (n).
        n0 = 1 / (snr_lin * rate)  # Noise variance

        # Convertir x en tenseur PyTorch s'il est sous forme NumPy
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

    
    def awgn(self):        # x : signal/message encodé
        # Finally calculate the variance of the AWGN
        n0 = 1/(self.snr_lin*self.rate)
        var_channel = math.sqrt(n0/2)

        # Use the reparametrization trick to apply noise to x
        if torch.is_tensor(self.x):
            x_channel = self.x + var_channel*torch.randn_like(self.x)  # Génère un bruit gaussien de même forme que x
        else:
            x_channel = self.x + var_channel*np.random.randn(*self.x.shape)  # Génère un bruit gaussien pour x.
        
        return x_channel
    
    
