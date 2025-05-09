import numpy as np
import torch
import matplotlib.pyplot as plt
import math

from channel import channel
from utils import bler

# To do block encoding (Hamming)
from sk_dsp_comm import fec_block as block

def qpsk(m, n, snr_db, num_bits=10000, chann_type="AWGN"):
    """
    Simule une transmission QPSK et calcule le taux d'erreur binaire (BER).

    Paramètres :
        snr_db (float) : SNR en dB.
        num_bits (int) : Nombre de bits à transmettre (doit être pair).
        chann_type (str) : Type de canal ("AWGN", "Rayleigh", "Rician").

    Retourne :
        ber (float) : Taux d'erreur binaire.
    """

    # Calcul du Nombre de Bits Nécessaires pour représenter m messages différents
    k = int(math.log2(m))
    
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
    if chann_type == "AWGN":
        _, _, received_symbols, _, _, _ = channel(symbols, snr_db, chann_type="AWGN")

    elif chann_type == "Rayleigh":
        _, _, received_symbols, _, _, _ = channel(symbols, snr_db, chann_type="Rayleigh")

    elif chann_type == "Rician":
        _, _, received_symbols, _, _, _ = channel(symbols, snr_db, chann_type="Rician")

    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}")

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
ber_awgn = [qpsk_communication(snr, chann_type="AWGN") for snr in snr_values]
ber_rayleigh = [qpsk_communication(snr, chann_type="Rayleigh") for snr in snr_values]
ber_rician = [qpsk_communication(snr, chann_type="Rician") for snr in snr_values]

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

def bpsk(snr_db, num_bits=10000, chann_type="AWGN"):
    """
    Simulates a simple BPSK transmission system.

    Parameters:
        snr_db (float): SNR in dB.
        num_bits (int): Number of bits to transmit.
        chann_type (str): Channel type ("AWGN", "Rayleigh", "Rician").

    Returns:
        ber (float): Bit error rate.
    """
    # Generate random bits (0s and 1s)
    bits = np.random.randint(0, 2, num_bits)

    # BPSK modulation: 0 -> -1, 1 -> +1
    symbols = 2 * bits - 1  

    # Appliquer les effets du canal
    if chann_type == "AWGN":
        _, _, received_signal, _, _, _ = channel(symbols, snr_db, chann_type="AWGN")

    elif chann_type == "Rayleigh":
        _, _, received_signal, _, _, _ = channel(symbols, snr_db, chann_type="Rayleigh")

    elif chann_type == "Rician":
        _, _, received_signal, _, _, _ = channel(symbols, snr_db, chann_type="Rician")

    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}")

    # BPSK demodulation
    detected_bits = (received_signal > 0).astype(int)

    # Compute BER
    errors = np.sum(bits != detected_bits)
    ber = errors / num_bits
    return ber

"""
snr_values = np.arange(-4, 30, 2)
ber_awgn = [bpsk_communication(snr, chann_type="AWGN") for snr in snr_values]
ber_rayleigh = [bpsk_communication(snr, chann_type="Rayleigh") for snr in snr_values]
ber_rician = [bpsk_communication(snr, chann_type="Rician") for snr in snr_values]

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

def block_encoder(x, n, k):
    """
    This is going to be the definition of encoding using Hamming
    Args:
        x of shape (batch_size, k): Messages without encoding
        n (int): Length of the encoded messages
        k (int): Length of the actual messages
    Returns:
        y of shape (batch_size, n): Encoded messages with Hamming
    """
    # There is no need for encoding
    # Si n=k, il n'y a pas besoin d'ajouter de bits de parité, car les messages d'entrée sont déjà à leur longueur maximale.
    if n == k:
        # Return as float because that the way encoder.hamm_encoder returns it
        return x
    
    # We initialize the encoder with the number of parity bits that we need
    # According to doc from block.fec_hamming
    # Initialized with j. Where n = 2^j-1. k = n-j.
    encoder = block.FECHamming(n-k)   # Initialisation de l'encodeur Hamming
    """
    block.FECHamming(n-k) : Initialise un encodeur Hamming avec n-k
    n : Longueur totale du message encodé.
    k : Longueur des bits d'information (message).
    n-k : Nombre de bits de parité(de contole).
    """
    
    # Allocation de l'espace pour les résultats
    batch_size, _ = x.shape  ## batch_size : Nombre de messages dans le lot (exemple, 32 messages dans un batch).
    # Pré-allocation :
    ## Crée une matrice de zéros de taille (batch_size,n) pour stocker les messages encodés.
    encoding_results = np.zeros((batch_size, n), dtype=int)
    
    # Encodage des Messages
    for i, x_vec in enumerate(x):  # Boucle sur chaque message dans le batch 
        x_vec = x_vec.astype(int)
        encoding_results[i, :] = encoder.hamm_encoder(x_vec)
    
    return encoding_results


def block_decoder(y, n, k):
    """
    This is going to be the definition of decoding using Hamming
    Args:
        x of shape (batch_size, n): Encoded messages
        n (int): Length of the encoded messages
        k (int): Length of the actual messages
    Returns:
        y of shape (batch_size, k): Decoded messages with Hamming
    """
    # There is no need for decoding
    if n == k:
        # Return as float because that the way encoder.hamm_decoder returns it
        return y
    
    # We initialize the decoder with the number of parity bits that we need
    # According to doc from block.fec_hamming
    # Initialized with j. Where n = 2^j-1. k = n-j.
    decoder = block.FECHamming(n-k)

    # Vérification des données binaires
    assert np.all(np.isin(y, [0, 1]))

    # Convertir en tableau NumPy si nécessaire
    if torch.is_tensor(y):
        y_vec = y.cpu().numpy()
    
    # Get the batch size and pre-allocate adequate space for it
    batch_size, _ = y.shape
    decoding_results = np.zeros((batch_size, k), dtype=int)
    
    # Iterate over the batches and get the encoding for all of them
    for i, y_vec in enumerate(y):
        y_vec = np.round(y_vec.cpu().numpy()).astype(int)  # Arrondir et convertir en int
        decoding_results[i, :] = decoder.hamm_decoder(y_vec)
    
    return decoding_results

def bpsk_communication(m, n, snr_db, n_blocks, chann_type = 'AWGN'):

    # Calcul du Nombre de Bits Nécessaires pour représenter m messages différents
    k = int(math.log2(m))

    # Génération des Messages
    """
    But : Créer des messages binaires (k bits par message) qui représentent les données à transmettre.
    Comment : Utilisation de la bibliothèque NumPy pour générer des séquences de bits aléatoires composées de 0 et 1.
    Cela génère une matrice où chaque ligne est un message de k bits.
    """
    x = np.random.randint(0, 2, size=(n_blocks, k))

    # Encodage des Messages avec Hamming
    """
    But : Ajouter des bits de redondance pour détecter et corriger des erreurs potentielles causées par le bruit du canal.
    Comment : Le code de Hamming (n,k) encode chaque message de k bits en une séquence de n bits. Par exemple, pour un code (7,4), 4 bits d’information sont codés en 7 bits, avec 3 bits ajoutés pour la détection/correction d’erreurs.
    
    the implementation of the Hamming algorithm can be found in Python library scikit-dsp-comm. 
    """
    x_encoded = block_encoder(x, n, k)  # Encode les messages x avec un code de Hamming (n,k)


    # Modulation BPSK

    """
    But : Convertir les bits encodés (0 et 1) en signaux modulés (-1 et +1) pour transmission.
    Comment : En BPSK, 0 est mappé à -1 et 1 est mappé à +1.
    """
    s_transmit = torch.tensor(2 * x_encoded - 1, dtype=torch.float32)

    # Ajout de Bruit via le Canal
    """
    But : Ajouter un bruit Gaussien au signal pour simuler les perturbations d'un canal réel (AWGN : Additive White Gaussian Noise).
    Comment : Générer du bruit avec une variance dépendant de la puissance du signal et du rapport signal/bruit (Eb/N0).
    """
    if chann_type == "AWGN":
        _, _, s_noise, _, _, _ = channel(s_transmit, snr_db, chann_type="AWGN")

    elif chann_type == "Rayleigh":
        _, _, s_noise, _, _, _ = channel(s_transmit, snr_db, chann_type="Rayleigh")

    elif chann_type == "Rician":
        _, _, s_noise, _, _, _ = channel(s_transmit, snr_db, chann_type="Rician")

    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}")

    # Démodulation BPSK
    """
    But : Récupérer les bits transmis (0 ou 1) à partir des signaux reçus (-1 ou +1).
    Comment : Utiliser un seuil pour décider si un signal correspond à un 0 ou à un 1. Si le signal > 0 ---> 1, sinon ---> 0.
    """
    y = np.sign(s_noise)
    y_enc = (y + 1) / 2

    # Décodage avec Hamming

    """
    But : Corriger les erreurs possibles dans les bits reçus en utilisant les bits de redondance.
    Comment : Appliquer l'algorithme de décodage de Hamming pour récupérer les k bits d'information.
    """ 
    x_rec = block_decoder(y_enc, n, k)


    # Calcul du BLER (Block Error Rate)
    """
    But : Mesurer la performance du système en calculant le pourcentage de blocs de données contenant des erreurs après décodage.
    Comment : Comparer les messages décodés aux messages originaux pour détecter les erreurs.
    """

    block_bler = bler(x, x_rec)

    print("Finished calculations for BPSK(%d, %d) channel: %s. SNR dB: %f." % (n, k, chann_type, snr_db))

    return block_bler

bpsk_communication(16, 7, 7, 10000, chann_type = 'AWGN')