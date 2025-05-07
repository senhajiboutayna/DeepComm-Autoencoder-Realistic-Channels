import torch
import numpy as np
from scipy.special import jv  # Fonction de Bessel pour le modèle de Jakes

def generate_time_varying_channel(model_type, n_symbols, fd_ts=0.1, rho=0.9, K_factor=0, seed=None):
    """
    Génère un canal variant dans le temps selon différents modèles.
    
    Args:
        model_type (str): Type de modèle de canal ("Jakes", "AR1", "static")
        n_symbols (int): Nombre de symboles/échantillons à générer
        fd_ts (float): Produit de la fréquence Doppler et période symbole (pour Jakes)
        rho (float): Coefficient de corrélation temporelle (pour AR1)
        K_factor (float): Facteur K pour canal de Rice (composante LOS)
        seed (int): Graine pour reproduction des résultats
        
    Returns:
        torch.Tensor: Coefficients du canal au cours du temps (n_symbols,)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if model_type == "static":
        # Canal statique (pour référence)
        if K_factor == 0:  # Rayleigh
            h_real = torch.randn(1) / np.sqrt(2)
            h_imag = torch.randn(1) / np.sqrt(2)
            h = torch.complex(h_real, h_imag).repeat(n_symbols)
        else:  # Rician
            s = np.sqrt(K_factor / (K_factor + 1))  # Composante LOS
            sigma = np.sqrt(1 / (2 * (K_factor + 1)))  # Composante NLOS
            h_real = s + sigma * torch.randn(1)
            h_imag = sigma * torch.randn(1)
            h = torch.complex(h_real, h_imag).repeat(n_symbols)
    
    elif model_type == "AR1":
        # Modèle autorégressif d'ordre 1 (AR1)
        h = torch.zeros(n_symbols, dtype=torch.complex64)
        
        # Initialisation
        if K_factor == 0:  # Rayleigh
            h_real = torch.randn(1) / np.sqrt(2)
            h_imag = torch.randn(1) / np.sqrt(2)
            h[0] = torch.complex(h_real, h_imag)
        else:  # Rician
            s = np.sqrt(K_factor / (K_factor + 1))  # Composante LOS
            sigma = np.sqrt(1 / (2 * (K_factor + 1)))  # Composante NLOS
            h_real = s + sigma * torch.randn(1)
            h_imag = sigma * torch.randn(1)
            h[0] = torch.complex(h_real, h_imag)
        
        # Générer le processus AR(1)
        for i in range(1, n_symbols):
            # Partie réelle
            noise_real = torch.randn(1) * np.sqrt((1 - rho**2) / 2)
            h_real = rho * h[i-1].real + noise_real
            
            # Partie imaginaire
            noise_imag = torch.randn(1) * np.sqrt((1 - rho**2) / 2)
            h_imag = rho * h[i-1].imag + noise_imag
            
            h[i] = torch.complex(h_real, h_imag)
    
    elif model_type == "Jakes":
        # Modèle de Jakes (basé sur la fonction d'autocorrélation)
        t = torch.arange(n_symbols, dtype=torch.float32)
        
        # Générer un processus complexe gaussien
        h_real = torch.randn(n_symbols) / np.sqrt(2)
        h_imag = torch.randn(n_symbols) / np.sqrt(2)
        h_complex = torch.complex(h_real, h_imag)
        
        # Calculer les coefficients de corrélation selon le modèle de Jakes
        R = torch.zeros(n_symbols)
        for i in range(n_symbols):
            # J0 est la fonction de Bessel d'ordre 0
            R[i] = jv(0, 2 * np.pi * fd_ts * i)
        
        # Appliquer la corrélation temporelle
        h = temporal_correlation(h_complex, R)
        
        # Ajouter composante LOS pour Rician si nécessaire
        if K_factor > 0:
            s = np.sqrt(K_factor / (K_factor + 1))
            h = h * np.sqrt(1 / (K_factor + 1)) + s
    
    else:
        raise ValueError(f"Type de modèle temporel non supporté: {model_type}")
    
    return h

def temporal_correlation(h_uncorrelated, R):
    """
    Applique une corrélation temporelle à un canal non corrélé.
    
    Args:
        h_uncorrelated (torch.Tensor): Coefficients du canal non corrélés
        R (torch.Tensor): Vecteur d'autocorrélation
        
    Returns:
        torch.Tensor: Canal avec corrélation temporelle
    """
    n = len(h_uncorrelated)
    h_fft = torch.fft.fft(h_uncorrelated)
    R_fft = torch.fft.fft(R, n=n)
    h_correlated = torch.fft.ifft(h_fft * torch.sqrt(R_fft.abs()))
    
    # Normaliser pour avoir une puissance moyenne de 1
    h_correlated = h_correlated / torch.sqrt(torch.mean(torch.abs(h_correlated)**2))
    
    return h_correlated

def time_varying_channel(x, h, snr_db, add_noise=True):
    """
    Applique un canal variant dans le temps à un signal d'entrée.
    
    Args:
        x (torch.Tensor): Signal d'entrée (n_symbols,)
        h (torch.Tensor): Coefficients du canal variant dans le temps (n_symbols,)
        snr_db (float): Rapport signal/bruit en dB
        add_noise (bool): Ajouter du bruit ou non
        
    Returns:
        torch.Tensor: Signal après passage dans le canal
        torch.Tensor: Coefficients du canal
    """
    # Vérifier les dimensions
    if len(x) != len(h):
        raise ValueError(f"Dimensions incompatibles: x({len(x)}) et h({len(h)})")
    
    # Appliquer le canal
    y = x * h
    
    # Ajouter du bruit si demandé
    if add_noise:
        snr_lin = 10**(snr_db / 10)
        noise_power = 1 / snr_lin
        noise = torch.sqrt(noise_power/2) * (torch.randn_like(y) + 1j * torch.randn_like(y))
        y = y + noise
    
    return y, h

def plot_time_varying_channel(h, model_type, fd_ts=None, rho=None):
    """
    Visualise un canal variant dans le temps.
    
    Args:
        h (torch.Tensor): Coefficients du canal
        model_type (str): Type de modèle de canal
        fd_ts (float): Produit de la fréquence Doppler et période symbole (pour Jakes)
        rho (float): Coefficient de corrélation temporelle (pour AR1)
    """
    import matplotlib.pyplot as plt
    
    n_symbols = len(h)
    t = np.arange(n_symbols)
    
    plt.figure(figsize=(12, 8))
    
    # Afficher le module du canal
    plt.subplot(2, 2, 1)
    plt.plot(t, torch.abs(h).numpy())
    plt.title(f"Module du canal ({model_type})")
    plt.xlabel("Symbole")
    plt.ylabel("|h|")
    plt.grid(True)
    
    # Afficher la phase du canal
    plt.subplot(2, 2, 2)
    plt.plot(t, torch.angle(h).numpy())
    plt.title(f"Phase du canal ({model_type})")
    plt.xlabel("Symbole")
    plt.ylabel("∠h (rad)")
    plt.grid(True)
    
    # Afficher les composantes réelle et imaginaire
    plt.subplot(2, 2, 3)
    plt.plot(t, h.real.numpy(), label='Réel')
    plt.plot(t, h.imag.numpy(), label='Imaginaire')
    plt.title(f"Composantes du canal ({model_type})")
    plt.xlabel("Symbole")
    plt.legend()
    plt.grid(True)
    
    # Afficher l'autocorrélation
    plt.subplot(2, 2, 4)
    max_lag = min(n_symbols // 2, 100)  # Limiter pour plus de clarté
    acorr = np.zeros(max_lag)
    h_np = h.numpy()
    
    for lag in range(max_lag):
        acorr[lag] = np.abs(np.correlate(h_np[lag:], h_np[:-lag if lag > 0 else None], mode='valid')[0]) / n_symbols
    
    plt.plot(np.arange(max_lag), acorr)
    
    # Ajouter la théorie pour comparaison
    if model_type == "Jakes" and fd_ts is not None:
        lags = np.arange(max_lag)
        theoretical_corr = jv(0, 2 * np.pi * fd_ts * lags)
        plt.plot(lags, theoretical_corr, 'r--', label='Théorique (Jakes)')
    elif model_type == "AR1" and rho is not None:
        lags = np.arange(max_lag)
        theoretical_corr = rho ** lags
        plt.plot(lags, theoretical_corr, 'r--', label='Théorique (AR1)')
    
    plt.title("Autocorrélation du canal")
    plt.xlabel("Décalage (symboles)")
    plt.ylabel("Autocorrélation")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def doppler_spectrum(h, fs=1.0):
    """
    Calcule et affiche le spectre Doppler du canal.
    
    Args:
        h (torch.Tensor): Coefficients du canal
        fs (float): Fréquence d'échantillonnage (symboles/seconde)
    """
    import matplotlib.pyplot as plt
    
    n_samples = len(h)
    
    # Calculer la FFT du canal
    h_fft = torch.fft.fft(h)
    h_fft_shifted = torch.fft.fftshift(h_fft)
    
    # Calculer les fréquences Doppler
    freq = torch.fft.fftshift(torch.fft.fftfreq(n_samples, d=1/fs))
    
    # Normaliser la puissance
    power_spectrum = torch.abs(h_fft_shifted)**2 / n_samples
    
    plt.figure(figsize=(10, 6))
    plt.plot(freq.numpy(), 10 * torch.log10(power_spectrum).numpy())
    plt.title("Spectre Doppler du canal")
    plt.xlabel("Fréquence Doppler (Hz)")
    plt.ylabel("Densité spectrale de puissance (dB)")
    plt.grid(True)
    plt.xlim(-fs/2, fs/2)
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    n_symbols = 1000
    
    # Générer un canal variant dans le temps avec différents modèles
    h_jakes = generate_time_varying_channel("Jakes", n_symbols, fd_ts=0.05)
    h_ar1 = generate_time_varying_channel("AR1", n_symbols, rho=0.98)
    h_static = generate_time_varying_channel("static", n_symbols)
    
    # Visualiser les canaux
    plot_time_varying_channel(h_jakes, "Jakes", fd_ts=0.05)
    plot_time_varying_channel(h_ar1, "AR1", rho=0.98)
    plot_time_varying_channel(h_static, "static")
    
    # Visualiser le spectre Doppler
    doppler_spectrum(h_jakes)
    
    # Test du canal avec un signal BPSK
    x = torch.tensor([1.0, -1.0] * (n_symbols // 2))
    y, h = time_varying_channel(x, h_jakes, snr_db=15)
    
    print(f"Puissance moyenne du canal: {torch.mean(torch.abs(h)**2)}")
    print(f"SNR en sortie: {10*torch.log10(torch.mean(torch.abs(y)**2) / torch.var(y-x*h)):.2f} dB")