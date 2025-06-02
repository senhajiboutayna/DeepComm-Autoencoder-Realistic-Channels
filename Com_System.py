import numpy as np
import matplotlib.pyplot as plt

def qpsk_communication(snr_db, num_bits=10000, channel_type="AWGN", K=3, plot_constellation=False):
    # Assurer que le nombre de bits est pair
    if num_bits % 2 != 0:
        num_bits += 1

    # Génération des bits aléatoires
    bits = np.random.randint(0, 2, num_bits)

    # Modulation QPSK
    bit_pairs = bits.reshape(-1, 2)
    symbols = (2 * bit_pairs[:, 0] - 1) + 1j * (2 * bit_pairs[:, 1] - 1)  
    symbols /= np.sqrt(2)  # Normalisation de la puissance

    # Convertir SNR de dB en échelle linéaire
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / (2 * snr_linear)  # Variance du bruit (correcte pour QPSK)
    noise = np.sqrt(noise_power) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))

    # Variables pour stocker les constellations
    received_no_eq = None
    received_with_eq = None

    # Appliquer les effets du canal
    if channel_type == "AWGN":
        received_symbols = symbols + noise

    elif channel_type == "Rayleigh":
        h = (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))) / np.sqrt(2)
        # Version sans égalisation
        received_no_eq = h * symbols + noise
        # Version avec égalisation MMSE
        received_with_eq = received_no_eq * np.conj(h) / (np.abs(h)**2 + noise_power)
        received_symbols = received_with_eq  # On utilise la version avec égalisation pour le BER

    elif channel_type == "Rician":
        h_los = np.ones(len(symbols))  # Composante LOS
        h_nlos = (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))) / np.sqrt(2)
        h = np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos
        received_symbols = h * symbols + noise
        # Égalisation MMSE
        received_symbols = received_symbols * np.conj(h) / (np.abs(h)**2 + noise_power)

    else:
        raise ValueError(f"Type de canal non supporté: {channel_type}")

    # Démodulation QPSK
    detected_bits = np.zeros((len(received_symbols), 2), dtype=int)
    detected_bits[:, 0] = np.real(received_symbols) > 0
    detected_bits[:, 1] = np.imag(received_symbols) > 0
    detected_bits = detected_bits.flatten()

    # Calcul du BER
    errors = np.sum(bits != detected_bits)
    ber = errors / num_bits

    # Visualisation des constellations si demandé et canal Rayleigh
    if plot_constellation and channel_type == "Rayleigh":
        plt.figure(figsize=(10, 6))
        
        # Constellation reçue sans égalisation
        plt.subplot(1, 2, 1)
        plt.scatter(np.real(received_no_eq), np.imag(received_no_eq), alpha=0.5)
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.5, c='r')
        plt.title(f'QPSK with Rayleigh Fading')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-3, 3); plt.ylim(-3, 3)
        
        # Constellation reçue avec égalisation
        plt.subplot(1, 2, 2)
        plt.scatter(np.real(received_with_eq), np.imag(received_with_eq), alpha=0.5)
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.5, c='r')
        plt.title(f'QPSK after MMSE Equalization')
        plt.xlabel('In-phase')
        plt.ylabel('Quadrature')
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5)
        
        plt.tight_layout()
        plt.savefig("plots/Rayleigh_Constellations.png")

    return ber 

def bpsk_communication(snr_db, num_bits=10000, channel_type="AWGN"):
    """
    Simulates a simple BPSK transmission system.

    Parameters:
        snr_db (float): SNR in dB.
        num_bits (int): Number of bits to transmit.
        channel_type (str): Channel type ("AWGN", "Rayleigh", "Rician").

    Returns:
        ber (float): Bit error rate.
    """
    # Generate random bits (0s and 1s)
    bits = np.random.randint(0, 2, num_bits)

    # BPSK modulation: 0 -> -1, 1 -> +1
    symbols = 2 * bits - 1  

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / (2 * snr_linear)  # AWGN noise variance
    noise = np.sqrt(noise_power) * np.random.randn(num_bits)  # Gaussian noise

    # Apply channel effects
    if channel_type == "AWGN":
        received_signal = symbols + noise

    elif channel_type == "Rayleigh":
        h = (np.random.randn(num_bits) + 1j * np.random.randn(num_bits)) / np.sqrt(2)
        received_signal = h * symbols + noise
        received_signal /= h  # Equalization

    elif channel_type == "Rician":
        K = 3  # Rician K-factor
        h_los = np.ones(num_bits)
        h_nlos = (np.random.randn(num_bits) + 1j * np.random.randn(num_bits)) / np.sqrt(2)
        h = np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos
        received_signal = h * symbols + noise
        received_signal /= h  # Equalization

    else:
        raise ValueError(f"Unsupported channel type: {channel_type}")

    # BPSK demodulation
    detected_bits = (received_signal > 0).astype(int)

    # Compute BER
    errors = np.sum(bits != detected_bits)
    ber = errors / num_bits
    return ber


"""
snr_values = np.arange(-4, 10, 2)

ber_awgn = [qpsk_communication(snr, channel_type="AWGN") for snr in snr_values]
ber_rayleigh = [qpsk_communication(snr, channel_type="Rayleigh") for snr in snr_values]
ber_rician = [qpsk_communication(snr, channel_type="Rician") for snr in snr_values]

plt.figure(figsize=(8, 5))
plt.semilogy(snr_values, ber_awgn, 'o-', label="AWGN")
plt.semilogy(snr_values, ber_rayleigh, 's-', label="Rayleigh")
plt.semilogy(snr_values, ber_rician, 'd-', label="Rician")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("Performance du système QPSK")
plt.legend()
plt.grid()

ber_awgn = [bpsk_communication(snr, channel_type="AWGN") for snr in snr_values]
ber_rayleigh = [bpsk_communication(snr, channel_type="Rayleigh") for snr in snr_values]
ber_rician = [bpsk_communication(snr, channel_type="Rician") for snr in snr_values]

plt.figure(figsize=(8, 5))
plt.semilogy(snr_values, ber_awgn, 'o-', label="AWGN")
plt.semilogy(snr_values, ber_rayleigh, 's-', label="Rayleigh")
plt.semilogy(snr_values, ber_rician, 'd-', label="Rician")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("Performance du système BPSK")
plt.legend()
plt.grid()
plt.show()
"""
