import numpy as np
import torch
import matplotlib.pyplot as plt

from channel import channel

def qpsk_communication(snr_db, num_bits=10000, channel_type="AWGN"):
    """
    Simule une transmission QPSK et calcule le taux d'erreur binaire (BER).

    Paramètres :
        snr_db (float) : SNR en dB.
        num_bits (int) : Nombre de bits à transmettre (doit être pair).
        channel_type (str) : Type de canal ("AWGN", "Rayleigh", "Rician").

    Retourne :
        ber (float) : Taux d'erreur binaire.
    """
    # Assurer que le nombre de bits est pair (chaque symbole QPSK = 2 bits)
    if num_bits % 2 != 0:
        num_bits += 1

    # Génération des bits aléatoires
    bits = np.random.randint(0, 2, num_bits)

    # Modulation QPSK (Mapper 2 bits → 1 symbole complexe)
    bit_pairs = bits.reshape(-1, 2)
    symbols = (2 * bit_pairs[:, 0] - 1) + 1j * (2 * bit_pairs[:, 1] - 1)  
    symbols /= np.sqrt(2)  # Normalisation de la puissance
    symbols = torch.tensor(symbols, dtype=torch.cfloat)

    # Appliquer les effets du canal
    if channel_type == "AWGN":
        _, _, received_symbols, _, _, _ = channel(symbols, snr_db, chann_type="AWGN")

    elif channel_type == "Rayleigh":
        _, _, received_symbols, _, _, _ = channel(symbols, snr_db, chann_type="Rayleigh")

    elif channel_type == "Rician":
        _, _, received_symbols, _, _, _ = channel(symbols, snr_db, chann_type="Rician")


    else:
        raise ValueError(f"Type de canal non supporté: {channel_type}")

    # Démodulation QPSK : Décision basée sur le quadrant
    detected_bits = np.zeros((len(received_symbols), 2), dtype=int)
    detected_bits[:, 0] = np.real(received_symbols) > 0  # Décision partie réelle
    detected_bits[:, 1] = np.imag(received_symbols) > 0  # Décision partie imaginaire
    detected_bits = detected_bits.flatten()  # Convertir en un tableau 1D

    # Calcul du BER
    errors = np.sum(bits != detected_bits)
    ber = errors / num_bits
    return ber 



"""
# Tracer BER vs SNR pour différents canaux
snr_values = np.arange(-4, 30, 2)
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
plt.show()
"""

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

    # Appliquer les effets du canal
    if channel_type == "AWGN":
        _, _, received_signal, _, _, _ = channel(symbols, snr_db, chann_type="AWGN")

    elif channel_type == "Rayleigh":
        _, _, received_signal, _, _, _ = channel(symbols, snr_db, chann_type="Rayleigh")

    elif channel_type == "Rician":
        _, _, received_signal, _, _, _ = channel(symbols, snr_db, chann_type="Rician")


    else:
        raise ValueError(f"Type de canal non supporté: {channel_type}")

    # BPSK demodulation
    detected_bits = (received_signal > 0).astype(int)

    # Compute BER
    errors = np.sum(bits != detected_bits)
    ber = errors / num_bits
    return ber

"""
snr_values = np.arange(-4, 30, 2)
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

