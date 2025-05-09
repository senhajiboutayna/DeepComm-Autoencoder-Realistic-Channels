import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import os

import matplotlib.pyplot as plt

from tqdm import tqdm

from channel import channel, feedback_csi
from models import Encoder, Decoder, FeedbackCorrection, Transmitter, Receiver
from utils import MemoryMessages, count_errors, bler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = True

# Créer un dossier pour sauvegarder les modèles
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

def save_models(encoder, decoder, feedback_model=None, prefix='', chann_type='AWGN'):
    torch.save(encoder.state_dict(), f'saved_models/{prefix}encoder_{chann_type}.pth', _use_new_zipfile_serialization=True)
    torch.save(decoder.state_dict(), f'saved_models/{prefix}decoder_{chann_type}.pth', _use_new_zipfile_serialization=True)
    if feedback_model is not None:
        torch.save(feedback_model.state_dict(), f'saved_models/{prefix}feedback_{chann_type}.pth', _use_new_zipfile_serialization=True)

def train_autoencoder(m, n, snr_db, chann_type, batch_size, n_epochs, lr, clipping, plot, stop_value, use_feedback = False, snr_feedback = None, compression_level = None, delay = None, sigma_CSI=0.5, binary=False, use_ml_feedback=False, use_robust_model=False):
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
    if use_robust_model:
        encoder = Transmitter(m=m, n=n, use_csi=use_feedback).to(device)
        decoder = Receiver(m=m, n=n, use_csi=use_feedback, use_ML=use_ml_feedback).to(device)
    else:
        encoder = Encoder(m=m, n=n).to(device)
        decoder = Decoder(m=m, n=n).to(device)

    # Modèle pour améliorer le feedback CSI
    feedback_model = None
    if use_feedback and use_ml_feedback:
        feedback_model = FeedbackCorrection(input_dim=n, hidden_dim=128, robust=use_robust_model).to(device)
        feedback_optimizer = optim.Adam(feedback_model.parameters(), lr=lr)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    # Learning rate scheduler
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', patience=100)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', patience=100)

    losses = []
    errors = []
    feedback_losses = []

    for epoch in tqdm(range(n_epochs), desc="Training"):
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

            data = torch.from_numpy(batch).long().to(device)  # [batch_size]
            targets = torch.from_numpy(targets_np).long().to(device)

            if use_robust_model:
                data_onehot = F.one_hot(data, num_classes=m).float().to(device)
                h_dummy = torch.zeros(data_onehot.shape[0], 2).to(device)
                encoded_data = encoder(data_onehot, h=h_dummy)
            else:
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
            _, data_channel, _, _, h_true, _ = channel(encoded_data, snr_db, chann_type=chann_type, sigma_CSI=current_sigma_CSI)
            data_channel = torch.clamp(data_channel, -clipping, clipping)
            h_input = h_true.to(device)

            # Décodage
            if use_robust_model:
                decoded_data = decoder(data_channel, h = h_input)
            else:
                decoded_data = decoder(data_channel)

            # Calcul de la perte principale
            targets = torch.from_numpy(targets_np).to(device).type(torch.long)
            main_loss = F.cross_entropy(decoded_data, targets)
            
            # Calcul de la perte de feedback
            if use_feedback and use_ml_feedback:
                feedback_loss = F.mse_loss(feedback_csi_value, h_true)
                total_loss = main_loss + 0.1*feedback_loss
                epoch_feedback_loss += feedback_loss.item()
            else :
                total_loss = main_loss

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            if use_feedback and use_ml_feedback:
                torch.nn.utils.clip_grad_norm_(feedback_model.parameters(), max_norm=1.0)

            encoder_optimizer.step()
            decoder_optimizer.step()
            if use_feedback and use_ml_feedback:
                feedback_optimizer.step()

            epoch_losses.append(main_loss.item())
            epoch_errors += count_errors(decoded_data, targets)
        
        # Mise à jour des schedulers
        encoder_scheduler.step(np.mean(epoch_losses))
        decoder_scheduler.step(np.mean(epoch_losses))

        # Enregistrement des métriques
        losses.append(np.mean(epoch_losses))
        errors.append(epoch_errors / m)
        if use_feedback and use_ml_feedback:
            feedback_losses.append(epoch_feedback_loss / m)

        # Affichage de l'avancement
        if epoch % plot == 0:
            log_str = f"Epoch {epoch}: Loss={losses[-1]:.4f}, BER={errors[-1]:.4f}"
            if use_ml_feedback:
                log_str = f"Epoch {epoch}: Loss={losses[-1]:.4f}, BER={errors[-1]:.4f}, Feedback Loss={feedback_losses[-1]:.4f}"
            print(log_str)

        # Critère d'arrêt précoce
        last_losses = np.array(losses[-10:])
        if np.all(last_losses < stop_value):
            print(f"Modèle convergé après {epoch} epochs.")
            break

    return encoder, decoder, feedback_model, errors, feedback_losses

chann_type = "Rayleigh"
use_robust_model = False
n_epochs = 20000
batch_size = 64
lr = 0.001
snr_db = 5  
clipping = 0.5

# Entraînement classique (sans feedback)
print("1. Training with perfect CSI...")
encoder_perfect, decoder_perfect, _, errors_perfect, _ = train_autoencoder(m=16, n=7, snr_db=snr_db, chann_type=chann_type, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                    clipping=clipping, plot=100, stop_value=0.0001, sigma_CSI=0.0, use_feedback=False, use_robust_model=use_robust_model)

save_models(encoder_perfect, decoder_perfect, prefix='perfect_', chann_type=chann_type)

# Entraînement avec Feedback bruité sans correction ML
print("\n2. Training with noisy feedback (no ML)...")
encoder_feedback, decoder_feedback, _, errors_feedback, _ = train_autoencoder(m=16, n=7, snr_db=snr_db ,chann_type=chann_type, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                                clipping=clipping, plot=100, stop_value=0.0001, sigma_CSI=0.5, use_feedback=True,
                                                                snr_feedback=7, compression_level=4, delay=2, binary=False, use_ml_feedback=False, use_robust_model=use_robust_model)

save_models(encoder_feedback, decoder_feedback, prefix='noisy_', chann_type=chann_type)

print("\n3. Training with noisy feedback (with ML)...")
encoder_ml, decoder_ml, feedback_model, errors_ml, feedback_losses = train_autoencoder(16, 7, snr_db=snr_db, chann_type=chann_type, batch_size=batch_size, n_epochs=n_epochs, lr=lr, 
                                                                                       clipping=clipping, plot=100, stop_value=0.0001, sigma_CSI=0.5, use_feedback=True,
                                                                                       snr_feedback=7, compression_level=4, delay=2, binary=False, use_ml_feedback=True, use_robust_model=use_robust_model)

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
plt.savefig(f"plots/{chann_type}_BER_epochs.png")

if feedback_model is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(feedback_losses, label="Perte du modèle de feedback")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Évolution de la perte du modèle de feedback")
        plt.legend()
        plt.grid()
        plt.savefig(f"plots/{chann_type}_loss.png")

plt.tight_layout()
plt.show()
