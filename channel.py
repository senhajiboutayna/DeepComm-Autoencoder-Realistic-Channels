import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, rayleigh, rice

def channel(x, snr_db, chann_type="AWGN", K_rician=3, sigma_CSI=0.0):
    """
    Simule un canal de communication avec AWGN, Rayleigh ou Rician fading.
    Simule un canal de communication avec CSI parfait ou bruité.
    
    Args:
        x : Signal d'entrée (PyTorch Tensor)
        snr_db : Rapport Signal/Bruit en dB
        chann_type : Type de canal ("AWGN", "Rayleigh" ou "Rician")
        K_rician : Facteur K du canal Rician (par défaut 3)
        sigma_CSI : Intensité du bruit sur l'estimation du canal (0 = CSI parfait)

    Returns:
        x_channel : Signal reçu après le canal
        x_received : Signal avec CSI parfait
        x_received_csi_bruite : Signal avec CSI bruité
        h : Véritable coefficient du canal
        h_hat : Estimation bruitée du canal
    """
    snr_lin = 10**(snr_db / 10)  # Convertir SNR dB en linéaire
    n0 = 1 / snr_lin  # Variance du bruit (normalisée)
    sigma_noise = np.sqrt(n0 / 2)  # Écart-type du bruit

    if chann_type == "AWGN":
        h = torch.ones_like(x)  # 🚀 Canal AWGN = pas d'effet de fading, donc h = 1
        noise = sigma_noise * torch.randn_like(x)  # Bruit Gaussien
        x_channel = h * x + noise
        print(f"Mean of noise - AWGN: {torch.mean(noise)}, Std of noise - AWGN: {torch.std(noise)}")

    elif chann_type == "Rayleigh":
        # Fading Rayleigh (module d'un signal complexe gaussien)
        h = torch.sqrt(torch.randn_like(x) ** 2 + torch.randn_like(x) ** 2) / np.sqrt(2)
        noise = sigma_noise * torch.randn_like(x)
        x_channel = h * x + noise  # Application du fading
        print(f"Mean x_channel - Rayleigh: {torch.mean(x_channel)}, Std x_channel - Rayleigh: {torch.std(x_channel)}")

    elif chann_type == "Rician":
        # Fading Rician = Composante directe + diffusion (Rayleigh)
        K = torch.tensor(K_rician, dtype=torch.float32)  # Facteur K
        s = np.sqrt(K)  # Composante directe (LOS)
        sigma_fading = np.sqrt(1 / (2 * (K + 1)))  # Composante diffusée (NLOS)

         # Génération du coefficient de fading Rician
        h_real = s + sigma_fading * torch.randn_like(x)
        h_imag = sigma_fading * torch.randn_like(x)
        h = torch.sqrt(h_real**2 + h_imag**2)  # Module du canal
    
        noise = sigma_noise * torch.randn_like(x)   # Bruit Gaussien
        x_channel = h * x + noise   # Application du fading
        print(f"Mean x_channel - Rician: {torch.mean(x_channel)}, Std x_channel - Rician: {torch.std(x_channel)}")

    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}")
    
    # Ajout du bruit sur l'estimation du canal (CSI imparfait)
    if chann_type == "AWGN":
        h_hat = 1 + sigma_CSI * 0.1 * torch.randn_like(h)  # Réduit l'impact du bruit
        h_hat = torch.clamp(h_hat, min=0.5, max=1.5)  # Garde une variation plus réaliste
        x_channel_CSI = h_hat * x + noise  # Signal reçu avec bruit sur l'estimation du canal
    else:
        h_hat = torch.clamp(h + sigma_CSI * torch.abs(torch.randn_like(h)), min=0.1)
        x_channel_CSI = h_hat * x + noise

    # Égalisation avec CSI parfait
    x_received = x_channel / h  

    # Égalisation avec CSI bruité
    x_received_CSI = x_channel / h_hat 

    return x_channel,x_channel_CSI, x_received, x_received_CSI, h, h_hat


def plot_channel_distribution(snr_db=10, n_samples=10000, chann_type="AWGN", K_rician=3):
    """
    Affiche l'histogramme du signal reçu après passage dans le canal.
    """
    x = torch.ones(n_samples)  # Signal d'entrée constant (1) pour bien voir l'effet du canal
    x_channel, _, _, _, _, _ = channel(x, snr_db, chann_type, K_rician)

    x_channel_np = x_channel.numpy()

    # Tracé de l'histogramme
    plt.figure(figsize=(8, 5))
    plt.hist(x_channel_np, bins=50, density=True, alpha=0.6, label="Simulated")

    # Théorie
    x_range = np.linspace(np.percentile(x_channel_np, 1), np.percentile(x_channel_np, 99), 1000)
    
    if chann_type == "AWGN":
        plt.plot(x_range, norm.pdf(x_range, loc=1, scale=np.sqrt(1 / (2 * snr_db))), 'r-', label="Theoretical")
    elif chann_type == "Rayleigh":
        plt.plot(x_range, rayleigh.pdf(x_range, scale=1 / np.sqrt(2)), 'r-', label="Theoretical")
    elif chann_type == "Rician":
        #v = np.sqrt(K_rician)  # Décalage théorique correct
        b = np.sqrt(6 * K_rician)
        sigma = 1 / np.sqrt(6)  # Échelle correcte 
        plt.plot(x_range, rice.pdf(x_range, b, scale=sigma), 'r-', label="Theoretical")

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

def evaluate_CSI_impact():
    """
    Évalue l'impact du CSI imparfait à travers plusieurs tests :
    - Variation du SNR
    - Variation du bruit sur CSI
    - Comparaison de la distribution de h et h_hat
    - Test sur différents types de canaux
    """
    snr = 20  # Valeur fixe du SNR
    sigma_CSI_values = [0.1, 0.5, 1.0]  # Bruit sur CSI
    channel_types = ["AWGN", "Rayleigh", "Rician"]

    x = torch.ones(10000)

    for chann_type in channel_types:
        plt.figure(figsize=(8, 5))

        for sigma_CSI in sigma_CSI_values:
            _, _, _, _, h, h_hat = channel(x, snr, chann_type, K_rician=3, sigma_CSI=sigma_CSI)
            plt.hist(h_hat.numpy(), bins=50, density=True, alpha=0.5, label=f"CSI bruité (σ={sigma_CSI})")

        plt.hist(h.numpy(), bins=50, density=True, alpha=0.5, label="h (vrai canal)")
        plt.xlabel("Valeur du coefficient de canal")
        plt.ylabel("Densité")
        plt.title(f"Distribution de h et h_hat ({chann_type}, SNR={snr} dB)")
        plt.legend()
        plt.grid()
        plt.show()

        # Affichage de la distribution du signal après le CSI
        plot_channel_distribution_CSI(x, snr, chann_type, K_rician=3, sigma_CSI=1.0)

def plot_channel_distribution_CSI(x, snr_db, chann_type="AWGN", K_rician=3, sigma_CSI=1.0):
    """
    Affiche l'histogramme du signal reçu après passage dans le canal, avec CSI bruité.
    """
    x_channel, x_channel_csi, x_received, x_received_CSI, _, _ = channel(x, snr_db, chann_type, K_rician, sigma_CSI)

    x_channel_np = x_channel.numpy()
    x_channel_csi_np = x_channel_csi.numpy()
    x_received_np = x_received.numpy()
    x_received_CSI_np = x_received_CSI.numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(x_channel_np, bins=50, density=True, alpha=0.5, label="Avec CSI parfait")
    plt.hist(x_channel_csi_np, bins=50, density=True, alpha=0.5, label="Avec CSI bruité")

    # Théorie
    x_range = np.linspace(np.percentile(x_channel_np, 1), np.percentile(x_channel_np, 99), 1000)
    
    if chann_type == "AWGN":
        plt.plot(x_range, norm.pdf(x_range, loc=1, scale=np.sqrt(1 / (10 * snr_db))), 'r-', label="Theoretical")
    elif chann_type == "Rayleigh":
        plt.plot(x_range, rayleigh.pdf(x_range, scale=1 / np.sqrt(2)), 'r-', label="Theoretical")
    elif chann_type == "Rician":
        #v = np.sqrt(K_rician)  # Décalage théorique correct
        b = np.sqrt(8 * K_rician)
        sigma = 1 / np.sqrt(8)  # Échelle correcte
        plt.plot(x_range, rice.pdf(x_range, b, scale=sigma), 'r-', label="Theoretical")


    plt.xlabel("Valeur du signal")
    plt.ylabel("Densité")
    plt.title(f"Distribution du signal après le canal {chann_type} (CSI bruité)")
    plt.legend()
    plt.grid()
    plt.show()



# Affichage des distributions simulées et théoriques
#plot_channel_distribution(snr_db=10, chann_type="AWGN")
#plot_channel_distribution(snr_db=10, chann_type="Rayleigh")
#plot_channel_distribution(snr_db=10, chann_type="Rician", K_rician=3)

# Affichage des distributions de fading
#plot_fading_distributions()

#Test avec un CSI bruité
evaluate_CSI_impact()