import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, rayleigh, rice

def channel(x, snr_db, chann_type="AWGN", K_rician=3):
    """
    Simule un canal de communication avec AWGN, Rayleigh ou Rician fading.
    
    Args:
        x : Signal d'entrée (PyTorch Tensor)
        snr_db : Rapport Signal/Bruit en dB
        chann_type : Type de canal ("AWGN", "Rayleigh" ou "Rician")
        K_rician : Facteur K du canal Rician (par défaut 3)

    Returns:
        x_channel : Signal reçu après le canal
    """
    snr_lin = 10**(snr_db / 10)  # Convertir SNR dB en linéaire
    n0 = 1 / snr_lin  # Variance du bruit (normalisée)
    sigma = np.sqrt(n0 / 2)  # Écart-type du bruit

    if chann_type == "AWGN":
        noise = sigma * torch.randn_like(x)  # Bruit Gaussien
        x_channel = x + noise
        print(f"Mean of noise - AWGN: {torch.mean(noise)}, Std of noise - AWGN: {torch.std(noise)}")

    elif chann_type == "Rayleigh":
        # Fading Rayleigh (module d'un signal complexe gaussien)
        h = torch.abs(torch.randn_like(x) + 1j * torch.randn_like(x)) / np.sqrt(2)
        noise = sigma * torch.randn_like(x)
        x_channel = h * x + noise  # Application du fading
        print(f"Mean x_channel - Rayleigh: {torch.mean(x_channel)}, Std x_channel - Rayleigh: {torch.std(x_channel)}")

    elif chann_type == "Rician":
        # Fading Rician = Composante directe + diffusion (Rayleigh)
        K = torch.tensor(K_rician, dtype=torch.float32)  # Facteur K
        s = np.sqrt(K / (K + 1))  # Composante directe (LOS)
        sigma_fading = np.sqrt(1 / (2 * (K + 1)))  # Composante diffusée (NLOS)

         # Génération du coefficient de fading Rician
        h_real = s + sigma_fading * torch.randn_like(x)
        h_imag = sigma_fading * torch.randn_like(x)
        h = torch.sqrt(h_real**2 + h_imag**2)  # Module du canal
    
        noise = sigma * torch.randn_like(x)   # Bruit Gaussien
        x_channel = h * x + noise   # Application du fading
        print(f"Mean x_channel - Rician: {torch.mean(x_channel)}, Std x_channel - Rician: {torch.std(x_channel)}")

    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}")

    return x_channel


def plot_channel_distribution(snr_db=10, n_samples=10000, chann_type="AWGN", K_rician=3):
    """
    Affiche l'histogramme du signal reçu après passage dans le canal.
    """
    x = torch.ones(n_samples)  # Signal d'entrée constant (1) pour bien voir l'effet du canal
    x_channel = channel(x, snr_db, chann_type, K_rician)

    x_channel_np = x_channel.numpy()

    # Tracé de l'histogramme
    plt.figure(figsize=(8, 5))
    plt.hist(x_channel_np, bins=50, density=True, alpha=0.6, label="Simulated")

    # Théorie
    x_range = np.linspace(0, x_channel_np.max(), 1000)
    
    if chann_type == "AWGN":
        plt.plot(x_range, norm.pdf(x_range, loc=1, scale=np.sqrt(1 / (2 * snr_db))), 'r-', label="Theoretical")
    elif chann_type == "Rayleigh":
        plt.plot(x_range, rayleigh.pdf(x_range, scale=1 / np.sqrt(2)), 'r-', label="Theoretical")
    elif chann_type == "Rician":
        #v = np.sqrt(K_rician)  # Décalage théorique correct
        sigma = 1 / np.sqrt(2)  # Échelle correcte
        plt.plot(x_range, rice.pdf(x_range, K_rician, scale=sigma), 'r-', label="Theoretical")

    plt.xlabel("Valeur du signal")
    plt.ylabel("Densité")
    plt.title(f"Distribution du signal après le canal {chann_type}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_fading_distributions():
    """
    Affichage des distributions Rayleigh et Rician pour différentes valeurs de K.
    """
    plt.figure(figsize=(8, 5))

    K_values = [0, 1, 2, 3, 4]  # Différentes valeurs de K
    x_range = np.linspace(0, 5, 500)

    for K in K_values:
        if K == 0:
            pdf = rayleigh.pdf(x_range, scale=1 / np.sqrt(2))
            label = "K=0 (Rayleigh)"
        else:
            pdf = rice.pdf(x_range, K, scale=1 / np.sqrt(2))
            label = f"K={K}"

        plt.plot(x_range, pdf, label=label)

    plt.xlabel("Valeur du signal")
    plt.ylabel("Densité")
    plt.title("Distributions Rayleigh et Rician")
    plt.legend()
    plt.grid()
    plt.show()


# Affichage des distributions simulées et théoriques
plot_channel_distribution(snr_db=10, chann_type="AWGN")
plot_channel_distribution(snr_db=10, chann_type="Rayleigh")
plot_channel_distribution(snr_db=10, chann_type="Rician", K_rician=3)

# Affichage des distributions de fading
plot_fading_distributions()