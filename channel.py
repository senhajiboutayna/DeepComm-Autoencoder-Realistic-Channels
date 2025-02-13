import torch
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


class Channel:
    def __init__(self, k, n, x, snr_db, chann_type):
        self.snr_db = snr_db
        self.chann_type = chann_type
        self.k = k
        self.n = n

        # Conversion dB -> linéaire
        self.snr_lin = 10 ** (snr_db / 10)
        self.rate = k / n  # Taux de codage
        self.n0 = 1 / (self.snr_lin * self.rate)  # Variance du bruit

        # Convertir x en tenseur PyTorch si nécessaire
        if not torch.is_tensor(x):
            self.x = torch.tensor(x, dtype=torch.float32)
        else:
            self.x = x

    def awgn(self):
        """ Ajoute un bruit gaussien au signal x """
        var_channel = math.sqrt(self.n0 / 2)

        # Génération du bruit gaussien
        if torch.is_tensor(self.x):
            x_channel = self.x + var_channel * torch.randn_like(self.x)
        else:
            x_channel = self.x + var_channel * np.random.randn(*self.x.shape)

        return x_channel

# Exemple de test
k, n = 4, 7
x = np.array([1, -1, 1, -1, 1, -1, 1])  # Signal encodé
snr_db = 10
chann_type = "AWGN"

channel = Channel(k, n, x, snr_db, chann_type)
x_noisy = channel.awgn()
print("Signal bruité :", x_noisy)