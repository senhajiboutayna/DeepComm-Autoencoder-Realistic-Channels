import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from channel import channel, feedback_csi
from models import Encoder, Decoder, FeedbackCorrection
from utils import count_errors, plot_constellations
from com_System import qpsk_communication

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_models(m, n, prefix='', chann_type='AWGN'):
    encoder = Encoder(m=m, n=n).to(device)
    decoder = Decoder(m=m, n=n).to(device)
    
    encoder.load_state_dict(torch.load(f'saved_models/{prefix}encoder_{chann_type}.pth'))
    decoder.load_state_dict(torch.load(f'saved_models/{prefix}decoder_{chann_type}.pth'))
    
    feedback_model = None
    if os.path.exists(f'saved_models/{prefix}feedback_model.pth'):
        feedback_model = FeedbackCorrection(input_dim=n, hidden_dim=128).to(device)
        feedback_model.load_state_dict(torch.load(f'saved_models/{prefix}feedback_{chann_type}.pth'))
    
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
                print(f"Avertissement: Tronquage de {num_symbols % k} éléments pour correspondre à k={k}")
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

# Charger les modèles sauvegardés
print("Chargement des modèles...")
m, n, k = 16, 7, 4
chann_type = 'Rayleigh'

encoder_perfect, decoder_perfect, _ = load_models(m, n, prefix='perfect_', chann_type=chann_type)
encoder_feedback, decoder_feedback, _ = load_models(m, n, prefix='noisy_', chann_type=chann_type)
encoder_ml, decoder_ml, feedback_model = load_models(m, n, prefix='ml_', chann_type=chann_type)

# Paramètres d'évaluation
snr_values = np.arange(-5, 10, 2)
n_samples = 20000

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
        'snr_feedback': 7, 
        'compression_level': 4, 
        'delay': 2
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
plt.semilogy(snr_values, results['perfect']['ber'], 'b-o', label='CSI parfait')
plt.semilogy(snr_values, results['noisy']['ber'], 'r--s', label='feedback bruité (sans ML)')
plt.semilogy(snr_values, results['ml']['ber'], 'g-.d', label='feedback bruité (avec ML)')
plt.semilogy(snr_values, ber_qpsk, 'c', label='QPSK')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('Comparaison des performances de transmission : BER')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Tracer Capacité du canal vs SNR
plt.figure(figsize=(10,6))
plt.plot(snr_values, results['perfect']['capacity'], 'b-o', label='CSI parfait')
plt.plot(snr_values, results['noisy']['capacity'], 'r--s', label='feedback bruité (sans ML)')
plt.plot(snr_values, results['ml']['capacity'], 'g-.d', label='feedback bruité (avec ML)')
plt.xlabel('SNR (dB)')
plt.ylabel('Capacité (bits/s/Hz)')
plt.title('Capacité théorique du canal en fonction du SNR')
plt.grid(True)
plt.legend()
plt.show()

# Tracer Latence de transmission vs SNR
plt.figure(figsize=(10,6))
plt.plot(snr_values, np.array(results['perfect']['latency'])*1000, 'b-o', label='CSI parfait')
plt.plot(snr_values, np.array(results['noisy']['latency'])*1000, 'r--s', label='feedback bruité (sans ML)')
plt.plot(snr_values, np.array(results['ml']['latency'])*1000, 'g-.d', label='feedback bruité (avec ML)')
plt.xlabel('SNR (dB)')
plt.ylabel('Latence (ms)')
plt.title('Latence de transmission en fonction du SNR')
plt.grid(True)
plt.legend()
plt.show()

# Visualisation des constellations
plot_constellations(
    results['perfect']['constellations'], 
    results['noisy']['constellations'], 
    results['ml']['constellations'],
    modulation='qpsk'
)
