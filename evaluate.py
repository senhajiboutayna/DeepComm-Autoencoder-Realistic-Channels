import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math

import matplotlib.pyplot as plt
from IPython.utils import io
import os
import time

from channel import channel, feedback_csi
from models import Encoder, Decoder, FeedbackCorrection
from utils import MemoryMessages, count_errors, plot_constellations
from com_System import qpsk_communication, bpsk_communication

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_models(m, n, prefix='', chann_type='AWGN', use_csi=False):
    encoder = Encoder(m=m, n=n).to(device)
    decoder = Decoder(m=m, n=n).to(device)
    
    encoder.load_state_dict(torch.load(f'saved_models/{prefix}encoder_{chann_type}.pth', weights_only=True))
    decoder.load_state_dict(torch.load(f'saved_models/{prefix}decoder_{chann_type}.pth', weights_only=True))
    
    feedback_model = None
    if os.path.exists(f'saved_models/{prefix}feedback_model.pth'):
        feedback_model = FeedbackCorrection(input_dim=n, hidden_dim=128, robust=False).to(device)
        feedback_model.load_state_dict(torch.load(f'saved_models/{prefix}feedback_{chann_type}.pth', weights_only=True))
    
    return encoder, decoder, feedback_model

def evaluate_autoencoder(encoder, decoder, m, n, k, snr_db, chann_type, n_samples, sigma_CSI=0.5, feedback_params=None, feedback_model=None):
    """
    Évalue les performances de l'autoencodeur en termes de taux d'erreur binaire (BER).

    Args:
        encoder (nn.Module): Le modèle de l'encodeur.
        decoder (nn.Module): Le modèle du décodeur.
        m (int): Nombre de messages possibles.
        n (int): Dimension du signal encodé.
        k (int): Nombre de bits par message.
        snr_db (float): Rapport signal sur bruit en dB.
        chann_type (str): Type de canal (par exemple, "AWGN").
        n_samples (int): Nombre d'échantillons à utiliser pour l'évaluation.
        sigma_CSI (float): Paramètre de variance pour le canal.
        feedback_params (dict): Paramètres du feedback (snr_feedback, compression_level, delay).
        feedback_model (nn.Module): Modèle pour améliorer le feedback CSI.

    Returns:
        dict: Résultats contenant BER, SER, capacité, latence et constellations.
    """
    encoder.eval()  # Mettre l'encodeur en mode évaluation
    decoder.eval()  # Mettre le décodeur en mode évaluation
    if feedback_model is not None:
        feedback_model.eval()

    metrics = {
        'ber' : 0,
        'ser' : 0,
        'capacity' : 0,
        'latency' : 0,
        'constellations' : []
    }

    start_time = time.time()  # Mesure du temps de transmission
    total_errors = 0
    total_symbol_errors = 0
    total_bits = 0

    with torch.no_grad():  # Désactiver le calcul du gradient pour l'évaluation
        for _ in range(n_samples):
            # Générer un message aléatoire
            message = np.random.randint(0, m, size=(n,))
            message_tensor = torch.from_numpy(message)
            message_tensor = message_tensor.unsqueeze(1)
            message_tensor = message_tensor.to(device)

            # Encoder le message
            encoded_data = encoder(message_tensor)

            # Gestion du feedback si activé
            current_sigma_CSI = sigma_CSI
            if feedback_params is not None:
                # Génération du vrai CSI (simulé)
                true_csi = torch.randn(encoded_data.shape, device=device)

                # Application du feedback avec ou sans ML
                feedback_csi_value = feedback_csi(true_csi, 
                                               feedback_params['snr_feedback'],
                                               feedback_params['compression_level'],
                                               feedback_params['delay'],
                                               binary=False,
                                               feedback_model=feedback_model,
                                               use_ml=(feedback_model is not None))
                current_sigma_CSI = feedback_csi_value

            # Passer le message encodé à travers le canal
            _, data_channel, _, _, _, _ = channel(encoded_data, snr_db, chann_type=chann_type, sigma_CSI=current_sigma_CSI)

            # Décoder le message
            decoded_data = decoder(data_channel)

            # Convertir la sortie du décodeur en prédiction
            predicted_message = torch.argmax(decoded_data, dim=1).cpu().numpy()

            # Compter les erreurs
            total_errors += np.sum(predicted_message != message)

            # Vérifier que la taille est un multiple de k
            num_symbols = predicted_message.shape[0]
            if num_symbols % k != 0:
                predicted_message = predicted_message[:num_symbols - (num_symbols % k)]
                message = message[:num_symbols - (num_symbols % k)]
            total_symbol_errors += np.sum(np.any(predicted_message.reshape(-1, k) != message.reshape(-1, k), axis=1))
            total_bits += k  # Chaque message contient k bits
            
            # Stockage des constellations pour visualisation
            if len(metrics['constellations']) < 1000: #Limiter le nmb stocké
                metrics['constellations'].append(data_channel.cpu().numpy())
    
    # Calcul de latence
    end_time = time.time()
    metrics['latency'] = end_time - start_time  # Latence de transmission

    # Calculer le BER et SER
    metrics['ber'] = total_errors / total_bits
    metrics['ser'] = total_symbol_errors / n_samples

    # Capacité du canal (Shannon)
    snr_linear = 10 ** (snr_db / 10)
    metrics['capacity'] = np.log2(1 + snr_linear)   # bits/s/Hz

    return metrics

snr_values = np.arange(-2, 11, 2)  # SNR en dB
n_samples = 30000  # Nombre d'échantillons pour l'évaluation
m, n = 16, 7  # Paramètres de l'autoencodeur
k = int(math.log2(m))
chann_type = 'Rayleigh'

encoder_perfect, decoder_perfect, _ = load_models(m, n, prefix='perfect_', chann_type=chann_type, use_csi=False)
encoder_feedback, decoder_feedback, _ = load_models(m, n, prefix='noisy_', chann_type=chann_type, use_csi=True)
encoder_ml, decoder_ml, feedback_model = load_models(m, n, prefix='ml_', chann_type=chann_type, use_csi=True)

# Stockage des résultats
results = {
    'perfect': {'ber': [], 'ser': [], 'capacity': [], 'latency': [], 'constellations': []},
    'noisy': {'ber': [], 'ser': [], 'capacity': [], 'latency': [], 'constellations': []},
    'ml': {'ber': [], 'ser': [], 'capacity': [], 'latency': [], 'constellations': []}
    }
ber_qpsk = []

# Boucle sur chaque SNR
for snr in snr_values:
    print(f"\nEvaluating SNR = {snr} dB...")

    # CSI parfait 
    print(" - Perfect CSI")
    metrics = evaluate_autoencoder(encoder_perfect, decoder_perfect, m, n, k, snr, chann_type=chann_type, n_samples=n_samples, sigma_CSI=0.0, feedback_params=None)
    for key in results['perfect']:
        results['perfect'][key].append(metrics[key])

    # Autoencodeur AVEC feedback bruité sans correction ML
    print(" - Noisy feedback (no ML)")
    feedback_params = {
        'snr_feedback': 10, 
        'compression_level': 2, 
        'delay': 1
    }
    metrics = evaluate_autoencoder(encoder_feedback, decoder_feedback, m, n, k, snr, chann_type=chann_type, n_samples=n_samples, sigma_CSI=0.5, feedback_params=feedback_params)
    for key in results['noisy']:
        results['noisy'][key].append(metrics[key])

    # Autoencodeur AVEC feedback bruité avec correction ML
    print(" - Noisy feedback (with ML)")
    metrics = evaluate_autoencoder(encoder_ml, decoder_ml, m, n, k, snr, chann_type=chann_type, n_samples=n_samples, sigma_CSI=0.5, feedback_params=feedback_params, feedback_model=feedback_model)
    for key in results['ml']:
        results['ml'][key].append(metrics[key])
    
    # QPSK
    ber_qpsk.append(qpsk_communication(snr_db=snr, num_bits=n_samples, channel_type=chann_type))

# Tracé BER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, results['perfect']['ber'], 'b-o', label='Perfect CSI')
plt.semilogy(snr_values, results['ml']['ber'], 'r--s', label='Noisy feedback (no corrected)')
plt.semilogy(snr_values, results['noisy']['ber'], 'g-.d', label='Noisy feedback (with ML correction)')
plt.semilogy(snr_values, ber_qpsk, 'c', label='QPSK')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('Comparison of transmission performance: BER for one channel ' + chann_type + ' channel')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig(f"plots/{chann_type}_BER.png")
plt.show()