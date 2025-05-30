import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import os

import matplotlib.pyplot as plt

from tqdm import tqdm

from channel import channel, feedback_csi
from models import Encoder, Decoder, FeedbackCorrection
from utils import MemoryMessages, count_errors, plot_constellation

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

def train_autoencoder(m, n, snr_db, chann_type, batch_size, n_epochs, lr, clipping, plot, stop_value, sigma_CSI = 0.5):

    
    k = math.log2(m)   # Get k Number of bits necessary to transmit the m messages

    # Initialize the encoder and decoder
    encoder = Encoder(m=m, n=n)
    encoder.to(device)
    decoder = Decoder(m=m, n=n)
    decoder.to(device)

    # Adam optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    # Variables pour suivre les pertes pendant l'entraînement
    losses = []   ## Cette liste losses est utilisée pour analyser ou tracer l'évolution de la perte au fil des itérations/époques.
    avg_losses = []
    errors = []   ## Suit le taux d'erreurs pour cette époque.
    avg_errors = []
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        message = MemoryMessages(m)
        epoch_losses = []  # Suivi des pertes pour cette époque
        epoch_errors = 0

        # Jusqu'à ce que nous ayons quelque chose dans la mémoire, l'époque n'est pas terminée.
        while len(message) > 0:
            batch , targets_np = message.sample(batch_size)
            # Make the gradients at the beginning 
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            data = torch.from_numpy(batch) #Conversion des Données NumPy vers Tenseur PyTorch
            data = data.unsqueeze(1) #Ajout de la dimension 1
            data = data.to(device) #Transfert du tenseur vers le GPU

            ### Passage du message par l'encoder
            encoded_data = encoder(data)

            # torch.isnan().any() : Détection de valeurs NaN dans un tenseur PyTorch
            if torch.isnan(encoded_data).any():
                print("NaN detected after encoder. Epoch: %d" % (epoch))
                break

            ### Passage du message par le canal 
            _, data_channel, _, _, _, _ = channel(encoded_data, snr_db, chann_type=chann_type, sigma_CSI=sigma_CSI)
            data_channel = torch.clamp(data_channel, -clipping, clipping)  # Ajustez les bornes si nécessaire
            if torch.isnan(data_channel).any():
                print("NaN detected after channel. Epoch: %d" % (epoch))
                break


            ### Passage du message par le decoder
            decoded_data = decoder(data_channel)
            if torch.isnan(decoded_data).any():
                print("NaN detected after Decoder. Epoch: %d" % (epoch))
                break

            # Conversion des targets en tenseur PyTorch 

            targets = torch.from_numpy(targets_np).to(device)
            targets = targets.type(torch.long)
            assert targets.dtype == torch.long, "Targets must be LongTensor"

            ### Calcul de la perte avec Cross Entropy
            loss = F.cross_entropy(decoded_data, targets, reduction='mean')
            loss.backward()
            # Vérification des NaN
            if torch.isnan(loss).any():
                print(f"NaN detected in loss at epoch {epoch}")

            # Ajouter la perte pour ce mini-lot
            epoch_losses.append(loss.item())

            # Compter le nombre d'erreurs
            epoch_errors += count_errors(decoded_data, targets)

            """
            Ces lignes effectuent une mise à jour des paramètres (poids et biais) des modèles d'encodeur (encoder) et de décodeur (decoder) 
            en utilisant l'algorithme d'optimisation choisi (ici, SGD, Adam ou autre).
            """
            encoder_optimizer.step()
            decoder_optimizer.step()

            losses.append(float(epoch_losses[-1]))
            errors.append(epoch_errors/m)

            #Récupère les dernières valeurs de perte enregistrées dans losses.
            """
            Cela permet d'évaluer si la perte diminue suffisamment sur les dernières itérations.
            Si la perte est constante ou très faible, cela peut indiquer que le modèle a convergé (il a appris aussi bien qu'il le peut).
            """
            last_losses = np.array(losses[-10:])
            if np.all(last_losses < stop_value):
                print("Le modele a converge après %d epochs." % (epoch))

            # If the loss is small enough the model has converged. Stop training
            if np.all(last_losses < stop_value):
                return encoder, decoder, errors
            
            """
        ce code se concentre sur la visualisation et le suivi des métriques pendant l'entraînement, comme la perte (loss) et les erreurs (errors)
        Objectif principal : Afficher et tracer les performances de l'autoencodeur pendant l'entraînement.
        """       

        # Affichage de l'avancement
        if epoch % plot == 0:
            log_str = f"Epoch {epoch}: Loss={losses[-1]:.4f}, Errors={errors[-1]:.4f}"
            print(log_str)

            # Calcul des Moyennes
            """
            Pourquoi les moyennes : 
            Les moyennes permettent de lisser les variations et de mieux observer les tendances.
            """

            if epoch > plot:
                avg = np.mean(losses[-plot:])
                avg_err = np.mean(errors[-plot:])
            else:
                avg = np.mean(losses)
                avg_err = np.mean(errors)
            avg_losses.append(avg)
            avg_errors.append(avg_err)

            """
            À la fin de l'entraînement, affiche les métriques finales:
            Dernière valeur des erreurs.
            Dernière valeur de la perte.

            Exemple d'affichage : 
            Finished training. Errors 0.020000. Loss: 0.000900
            """

            if epoch == n_epochs-1 : 
                print("Finished training. Errors %f. Loss: %f" % (errors[-1], losses[-1]))


    return encoder, decoder, errors

def train_autoencoder_with_feedback(m, n, snr_db, snr_feedback, compression_level, delay, chann_type, batch_size, n_epochs, lr, clipping, plot, stop_value, sigma_CSI=None, binary=False, use_ml_feedback=True):
    """
    Entraîne l'autoencodeur en utilisant un feedback CSI bruité et compressé.

    Args:
        m (int): Nombre de messages possibles.
        n (int): Dimension du signal encodé.
        snr_db (float): Rapport signal/bruit principal (canal direct).
        snr_feedback (float): Rapport signal/bruit du canal de feedback.
        compression_level (int): Niveau de compression du CSI feedback.
        delay (int): Délai du feedback CSI.
        chann_type (str): Type de canal (ex: "Rayleigh").
        batch_size (int): Taille du batch.
        n_epochs (int): Nombre d'époques d'entraînement.
        lr (float): Taux d'apprentissage.
        clipping (float): Valeur de clipping pour éviter les explosions de gradient.
        plot (int): Fréquence d'affichage des courbes d'entraînement.
        stop_value (float): Seuil d'arrêt basé sur la perte.
        sigma_CSI (float): Bruit sur l'estimation CSI.

    Returns:
        encoder, decoder, errors: Modèles entraînés et liste des erreurs.
    """

    k = math.log2(m)

    encoder = Encoder(m=m, n=n).to(device)
    decoder = Decoder(m=m, n=n, use_csi=True).to(device)

    # Modèle pour améliorer le feedback CSI
    feedback_model = None
    if use_ml_feedback:
        feedback_model = FeedbackCorrection(input_dim=n).to(device)
        feedback_optimizer = optim.Adam(feedback_model.parameters(), lr=lr)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

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
            targets = torch.from_numpy(targets_np).to(device).type(torch.long)
            data = torch.from_numpy(batch).unsqueeze(1).to(device)

            if use_ml_feedback:
                # Phase 1: Train feedback model alone
                with torch.no_grad():
                    encoded_data = encoder(data)
                    _, _, _, _, h_true, _ = channel(encoded_data, snr_db, chann_type, sigma_CSI=sigma_CSI)
                
                feedback_optimizer.zero_grad()
                feedback_pred = feedback_csi(h_true, snr_feedback, compression_level, delay, 
                                        binary, feedback_model, use_ml=True)
                feedback_loss = F.mse_loss(feedback_pred, h_true, reduction='mean')
                feedback_loss.backward()
                feedback_optimizer.step()
                epoch_feedback_loss += feedback_loss.item()
                
                # Phase 2: Train full system
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                encoded_data = encoder(data)
                feedback_csi_value = feedback_csi(h_true, snr_feedback, compression_level, 
                                            delay, binary, feedback_model, use_ml=True)
                
                # Channel with current feedback
                _, data_channel, _, _, _, _ = channel(encoded_data, snr_db, chann_type=chann_type, 
                                                    h_override=feedback_csi_value)
                data_channel = torch.clamp(data_channel, -clipping, clipping)
                
                decoded_data = decoder(data_channel, h=feedback_csi_value)
                loss = F.cross_entropy(decoded_data, targets)
                
                # Combined loss with stronger feedback weighting
                total_loss = loss + 0.5 * feedback_loss.detach()  # Increased feedback weight
                total_loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                
            else:
                # Original non-ML feedback training
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                encoded_data = encoder(data)
                _, _, _, _, h_true, _ = channel(encoded_data, snr_db, chann_type, sigma_CSI=sigma_CSI)
                feedback_csi_value = feedback_csi(h_true, snr_feedback, compression_level, 
                                            delay, binary, None, use_ml=False)
                
                _, data_channel, _, _, _, _ = channel(encoded_data, snr_db, chann_type=chann_type, 
                                                    sigma_CSI=feedback_csi_value)
                decoded_data = decoder(data_channel, h=feedback_csi_value)
                loss = F.cross_entropy(decoded_data, targets)
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

            epoch_losses.append(loss.item())
            epoch_errors += count_errors(decoded_data, targets)

        losses.append(np.mean(epoch_losses))
        errors.append(epoch_errors / m)
        if use_ml_feedback:
            feedback_losses.append(epoch_feedback_loss / m)
        
        # Early stopping
        if len(losses) > 20 and np.mean(losses[-20:]) < stop_value:
            print(f"\nEarly stopping at epoch {epoch}")
            break

        # Affichage de l'avancement
        if epoch % plot == 0:
            log_str = f"Epoch {epoch}: Loss={losses[-1]:.4f}, Errors={errors[-1]:.4f}"
            if use_ml_feedback:
                log_str += f", Feedback Loss={feedback_losses[-1]:.4f}"
            print(log_str)

    return encoder, decoder, feedback_model, errors, feedback_losses

chann_type = "Rayleigh"
m, n = 16, 7
n_epochs = 10000
batch_size = 64
lr = 0.0001
snr_db = 7  
clipping = 1e5

# Entraînement classique (sans feedback)
print("1. Training with perfect CSI - snr_db = ", snr_db)
encoder_perfect, decoder_perfect, errors_perfect = train_autoencoder(m=m, n=n, snr_db=snr_db, chann_type=chann_type, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                    clipping=clipping, plot=100, stop_value=0.000005, sigma_CSI=0.0)


save_models(encoder_perfect, decoder_perfect, prefix='perfect_', chann_type=chann_type)

# Entraînement avec Feedback bruité sans correction ML
print("\n2. Training with noisy feedback (no ML) - snr_db = ", snr_db)
encoder_feedback, decoder_feedback, _, errors_feedback, _ = train_autoencoder_with_feedback(m=m, n=n, snr_db=snr_db, snr_feedback=7, compression_level=4, delay=2,
                                                                chann_type=chann_type, batch_size=batch_size, n_epochs=n_epochs, lr=lr,clipping=clipping, 
                                                                plot=100, stop_value=0.000005, sigma_CSI=1.0, binary=False, use_ml_feedback=False)

save_models(encoder_feedback, decoder_feedback, prefix='noisy_', chann_type=chann_type)

print("\n3. Training with noisy feedback (with ML) - snr_db = ", snr_db)
encoder_ml, decoder_ml, feedback_model, errors_ml, feedback_losses = train_autoencoder_with_feedback(m=m, n=n, snr_db=snr_db, snr_feedback=7, compression_level=4, 
                                                                delay=2, chann_type=chann_type, batch_size=batch_size, n_epochs=n_epochs, lr=lr,clipping=clipping, 
                                                                plot=100, stop_value=0.0001, sigma_CSI=0.5, binary=False, use_ml_feedback=True)

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
