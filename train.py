import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import os

import matplotlib.pyplot as plt

from channel import channel, feedback_csi
from models import Encoder, Decoder, FeedbackCorrection, TemporalEncoder, TemporalDecoder
from utils import MemoryMessages, count_errors

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = True

# Créer un dossier pour sauvegarder les modèles
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

def save_models(encoder, decoder, feedback_model=None, prefix='', chann_type='AWGN'):
    torch.save(encoder.state_dict(), f'saved_models/{prefix}encoder_{chann_type}.pth')
    torch.save(decoder.state_dict(), f'saved_models/{prefix}decoder_{chann_type}.pth')
    if feedback_model is not None:
        torch.save(feedback_model.state_dict(), f'saved_models/{prefix}feedback_{chann_type}.pth')

def train_autoencoder(m, n, snr_db, chann_type, batch_size, n_epochs, lr, clipping, plot, stop_value, use_feedback = False, snr_feedback = None, compression_level = None, delay = None, sigma_CSI=0.5, binary=False, use_ml_feedback=False):
    """
    Entraîne l'autoencodeur en utilisant un feedback CSI bruité et compressé.

    Args:
        m (int): Nombre de messages possibles.
        n (int): Dimension du signal encodé.
        snr_db (float): Rapport signal/bruit principal (canal direct).
        chann_type (str): Type de canal (ex: "AWGN").
        batch_size (int): Taille du batch.
        n_epochs (int): Nombre d'époques d'entraînement.
        lr (float): Taux d'apprentissage.
        clipping (float): Valeur de clipping pour éviter les explosions de gradient.
        plot (int): Fréquence d'affichage des courbes d'entraînement.
        stop_value (float): Seuil d'arrêt basé sur la perte.
        use_feedback (bool): Utiliser le feedback CSI.
        snr_feedback (float): Rapport signal/bruit du canal de feedback.
        compression_level (int): Niveau de compression du CSI feedback.
        delay (int): Délai du feedback CSI.
        sigma_CSI (float): Bruit sur l'estimation CSI.
        binary (bool): Utiliser le feedback binaire.
        use_ml_feedback (bool): Utiliser le feedback ML.

    Returns:
        encoder, decoder, errors: Modèles entraînés et liste des erreurs.
    """

    k = math.log2(m)

    encoder = TemporalEncoder(m=m, n=n).to(device)
    decoder = TemporalDecoder(m=m, n=n).to(device)

    # Modèle pour améliorer le feedback CSI
    feedback_model = None
    if use_feedback and use_ml_feedback:
        feedback_model = FeedbackCorrection(input_dim=n, hidden_dim=128).to(device)
        feedback_optimizer = optim.Adam(feedback_model.parameters(), lr=lr)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    losses = []
    errors = []
    feedback_losses = []

    for epoch in range(n_epochs):
        message = MemoryMessages(m)
        epoch_losses = []
        epoch_errors = 0
        epoch_feedback_loss = 0

        while len(message) > 0:
            batch, targets_np = message.sample(batch_size)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            if use_feedback and use_ml_feedback:
                feedback_optimizer.zero_grad()

            data = torch.from_numpy(batch).unsqueeze(1).to(device)

            encoded_data = encoder(data)

            # Gestion du feedback
            current_sigma_CSI = sigma_CSI
            if use_feedback :
                with torch.no_grad():
                    _, _, _, _, h_true, _ = channel(encoded_data, snr_db, chann_type, sigma_CSI=sigma_CSI)
                feedback_csi_value = feedback_csi(
                    h_true, 
                    snr_feedback if use_feedback else 0, 
                    compression_level if use_feedback else 0, 
                    delay if use_feedback else 0, 
                    binary, 
                    feedback_model if use_ml_feedback else None, 
                    use_ml_feedback
                )
                current_sigma_CSI = feedback_csi_value
                
            # Passage par le canal    
            _, data_channel, _, _, _, _ = channel(encoded_data, snr_db, chann_type=chann_type, sigma_CSI=current_sigma_CSI)
            data_channel = torch.clamp(data_channel, -1e5, 1e5)

            # Décodage
            decoded_data = decoder(data_channel)

            # Calcul de la perte principale
            targets = torch.from_numpy(targets_np).to(device).type(torch.long)
            loss = F.cross_entropy(decoded_data, targets)
            
            # Calcul de la perte de feedback
            if use_feedback and use_ml_feedback:
                feedback_loss = F.mse_loss(feedback_csi_value, h_true)
                total_loss = loss + 0.1 *feedback_loss
                epoch_feedback_loss += feedback_loss.item()
            else :
                total_loss = loss

            total_loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            if use_feedback and use_ml_feedback:
                feedback_optimizer.step()

            epoch_losses.append(loss.item())
            epoch_errors += count_errors(decoded_data, targets)

        # Enregistrement des métriques
        losses.append(np.mean(epoch_losses))
        errors.append(epoch_errors / m)
        if use_feedback and use_ml_feedback:
            feedback_losses.append(epoch_feedback_loss / m)

        # Affichage de l'avancement
        if epoch % plot == 0:
            log_str = f"Epoch {epoch}: Loss={losses[-1]:.4f}, BER={errors[-1]:.4f}"
            if use_feedback and use_ml_feedback:
                log_str += f", Feedback Loss={feedback_losses[-1]:.4f}"
            print(log_str)

        # Vérification de la convergence
        last_losses = np.array(losses[-10:])
        if np.all(last_losses < stop_value):
            print(f"Modèle convergé après {epoch} epochs.")
            break

    return encoder, decoder, feedback_model, errors, feedback_losses

chann_type = "Rayleigh"
n_epochs = 20000

# Entraînement classique (sans feedback)
print("1. Training with perfect CSI...")
encoder_perfect, decoder_perfect, _, errors_perfect, _ = train_autoencoder(m=16, n=7, snr_db=7, chann_type=chann_type, batch_size=64, n_epochs=n_epochs, lr=0.001,
                                                    clipping=0.5, plot=100, stop_value=0.000005, sigma_CSI=0.0, use_feedback=False)

save_models(encoder_perfect, decoder_perfect, prefix='perfect_', chann_type=chann_type)

# Entraînement avec Feedback bruité sans correction ML
print("\n2. Training with noisy feedback (no ML)...")
encoder_feedback, decoder_feedback, _, errors_feedback, _ = train_autoencoder(m=16, n=7, snr_db=7,chann_type=chann_type, batch_size=64, n_epochs=n_epochs, lr=0.001,
                                                                clipping=0.5, plot=100, stop_value=0.000005, sigma_CSI=0.0, use_feedback=True,
                                                                snr_feedback=7, compression_level=4, delay=2, binary=False, use_ml_feedback=False)

save_models(encoder_feedback, decoder_feedback, prefix='noisy_', chann_type=chann_type)

print("\n3. Training with noisy feedback (with ML)...")
encoder_ml, decoder_ml, feedback_model, errors_ml, feedback_losses = train_autoencoder(16, 7, snr_db=7, chann_type=chann_type, batch_size=64, n_epochs=n_epochs, lr=0.001, 
                                                                                       clipping=0.5, plot=100, stop_value=0.0001, sigma_CSI=0.0, use_feedback=True,
                                                                                       snr_feedback=7, compression_level=4, delay=2, binary=False, use_ml_feedback=True)

save_models(encoder_ml, decoder_ml, feedback_model, prefix='ml_', chann_type=chann_type)


# Tracer les courbes d'entraînement
plt.figure(figsize=(8,5))
plt.plot(errors_perfect, label="Sans Feedback (CSI parfait)")
plt.plot(errors_feedback, label="Avec Feedback Bruité (sans ML)")
plt.plot(errors_ml, label="Avec Feedback Bruité (ML)")
plt.xlabel("Epochs")
plt.ylabel("BER")
plt.title("Impact du Feedback Bruité sur l'Autoencodeur")
plt.legend()
plt.grid()
plt.show()

if feedback_model is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(feedback_losses, label="Perte du modèle de feedback")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Évolution de la perte du modèle de feedback")
        plt.legend()
        plt.grid()
        plt.show()
