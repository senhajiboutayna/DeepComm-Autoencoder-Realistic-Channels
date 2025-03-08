import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math

import matplotlib.pyplot as plt
from IPython.utils import io
import time

from channel import channel
from models import Encoder, Decoder
from utils import MemoryMessages, count_errors, plot_constellation
from com_System import bpsk_communication, qpsk_communication

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = True

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
    
    for epoch in range(n_epochs):
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

            print(f"Data Shape: {data.shape}")  # Assurer que la forme est correcte

            ### Passage du message par l'encoder
            encoded_data = encoder(data)
            print(f"Encoded Data Shape: {encoded_data.shape}")  # Vérifie la sortie de l'encodeur

            # torch.isnan().any() : Détection de valeurs NaN dans un tenseur PyTorch
            if torch.isnan(encoded_data).any():
                print("NaN detected after encoder. Epoch: %d" % (epoch))
                break

            ### Passage du message par le canal 
            _, data_channel, _, _, _, _ = channel(encoded_data, snr_db, chann_type=chann_type, sigma_CSI=sigma_CSI)
            data_channel = torch.clamp(data_channel, -1e5, 1e5)  # Ajustez les bornes si nécessaire
            if torch.isnan(data_channel).any():
                print("NaN detected after channel. Epoch: %d" % (epoch))
                break

            print(f"Data Channel Shape: {data_channel.shape}")

            ### Passage du message par le decoder
            decoded_data = decoder(data_channel)
            if torch.isnan(decoded_data).any():
                print("NaN detected after Decoder. Epoch: %d" % (epoch))
                break

            print(f"Decoded Data Shape: {decoded_data.shape}")

            # Conversion des targets en tenseur PyTorch 

            targets = torch.from_numpy(targets_np).to(device)
            targets = targets.type(torch.long)
            print(f"Targets Shape: {targets.shape}")
            assert targets.dtype == torch.long, "Targets must be LongTensor"

            ### Calcul de la perte avec Cross Entropy
            loss = F.cross_entropy(decoded_data, targets)
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
            print('les 10 dernières valeurs de perte :', last_losses)
            if np.all(last_losses < stop_value):
                print("Le modele a converge après %d epochs." % (epoch))
            else:
                print("L'entrainement continue.")

            # If the loss is small enough the model has converged. Stop training
            if np.all(last_losses < stop_value):
                return encoder, decoder, errors
            
            """
        ce code se concentre sur la visualisation et le suivi des métriques pendant l'entraînement, comme la perte (loss) et les erreurs (errors)
        Objectif principal : Afficher et tracer les performances de l'autoencodeur pendant l'entraînement.
        """       

        if plot is not None : 
            #Affichage des Résultats de l’Époque : 
            print("Finished epoch: %d. Errors %f. Loss: %f" % (epoch, errors[-1], losses[-1]), end="\r")
            """ 
            Exemple d'affichage : 
            Finished epoch: 5. Errors 0.025000. Loss: 0.001234            
            """

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

            # Tracer les erreurs
            if  epoch%plot==0 and epoch != 0 :
                plt.clf() ## Efface l'ancien graphique pour éviter qu'il ne se superpose au nouveau.
                plt.plot(errors, label = 'Errors. Supervised training')
                plt.plot(avg_errors,label="Average Errors. Supervised training")
                plt.legend(loc = 'upper right')
                plt.draw()
                #plt.show()

            """
            À la fin de l'entraînement, affiche les métriques finales:
            Dernière valeur des erreurs.
            Dernière valeur de la perte.

            Exemple d'affichage : 
            Finished training. Errors 0.020000. Loss: 0.000900
            """

            if epoch == n_epochs-1 : 
                print("Finished training. Errors %f. Loss: %f" % (errors[-1], losses[-1]))

            print("Errors :", errors)


    return encoder, decoder, errors

def evaluate_autoencoder(encoder, decoder, m, n, k, snr_db, chann_type, n_samples, sigma_CSI=0.5):
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

    Returns:
        float: Le taux d'erreur binaire (BER) calculé.
    """
    encoder.eval()  # Mettre l'encodeur en mode évaluation
    decoder.eval()  # Mettre le décodeur en mode évaluation

    total_errors = 0
    total_symbol_errors = 0
    total_bits = 0
    received_symbols_list = []

    start_time = time.time()  # Mesure du temps de transmission

    with torch.no_grad():  # Désactiver le calcul du gradient pour l'évaluation
        for _ in range(n_samples):
            # Générer un message aléatoire
            message = np.random.randint(0, m, size=(n,))
            message_tensor = torch.from_numpy(message)
            message_tensor = message_tensor.unsqueeze(1)
            message_tensor = message_tensor.to(device)

            # Encoder le message
            encoded_data = encoder(message_tensor)

            # Passer le message encodé à travers le canal
            _, data_channel, _, _, _, _ = channel(encoded_data, snr_db, chann_type=chann_type, sigma_CSI=sigma_CSI)

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
            received_symbols_list.append(data_channel.cpu().numpy())
    
    end_time = time.time()
    latency = end_time - start_time  # Latence de transmission
    print("Latency:", latency, "seconds")

    # Calculer le BER
    ber = total_errors / total_bits
    ser = total_symbol_errors / n_samples

    # Capacité du canal (Shannon)
    snr_linear = 10 ** (snr_db / 10)
    capacity = np.log2(1 + snr_linear)   # bits/s/Hz

    return ber, snr_db, ser, capacity, latency, received_symbols_list
    

def plot_training_loss(losses):
    """
    Trace l'évolution de la perte pendant l'entraînement.
    """
    plt.figure(figsize=(8,5))
    plt.plot(losses, label="Perte d'entraînement")
    plt.xlabel("Itérations")
    plt.ylabel("Perte (Cross Entropy)")
    plt.title("Évolution de la perte pendant l'entraînement")
    plt.legend()
    plt.grid()
    plt.show()

if train:
    encoder, decoder, errors = train_autoencoder(m=16, n=7,snr_db=7 ,chann_type="Rayleigh", batch_size=64, n_epochs=10000, lr=0.001,
                                clipping=0.5, plot=10, stop_value=0.000005, sigma_CSI=0.0)    
    
    plot_training_loss(errors)  # Affichage de l'évolution de la perte 


# Évaluation de l'autoencodeur
snr_values = np.arange(-4, 20, 1)  # Exemple de valeurs de SNR
ber_autoencoder_list = []
ser_autoencoder_list = []
capacity_autoencoder_list = []
latency_autoencoder_list = []
ber_bpsk_list = []
ber_qpsk_list = []

for snr in snr_values:
    # Évaluation de l'autoencodeur
    ber_autoencoder, snr_db, ser, capacity, latency, received_symbols = evaluate_autoencoder(encoder, decoder, m=16, n=7, k=4, snr_db=snr, chann_type="Rayleigh", n_samples=10000, sigma_CSI=0.0)
    print(f"BER de l'autoencodeur (SNR= {snr_db} dB): {ber_autoencoder:.6f}")
    ber_autoencoder_list.append(ber_autoencoder)
    ser_autoencoder_list.append(ser)
    capacity_autoencoder_list.append(capacity)
    latency_autoencoder_list.append(latency)
    
    # Évaluation du système BPSK
    # Vous devez ajuster le code BPSK pour accepter un SNR variable
    ber_bpsk = bpsk_communication(snr_db=snr,num_bits=10000, channel_type="Rayleigh")
    ber_bpsk_list.append(ber_bpsk)

    ber_qpsk = qpsk_communication(snr_db=snr,num_bits=10000, channel_type="Rayleigh")
    ber_qpsk_list.append(ber_qpsk)

# Tracer BER and SER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, ber_autoencoder_list, 'b', label='BER (Autoencodeur)')
plt.semilogy(snr_values, ser_autoencoder_list, 'm', label='SER (Autoencodeur)')
plt.semilogy(snr_values, ber_bpsk_list, 'r', label='BER (BPSK)')
plt.semilogy(snr_values, ber_qpsk_list, 'g', label='BER (QPSK)')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('Comparaison des Performances entre Autoencodeur et QPSK')
plt.grid(True)
plt.legend()
plt.show()

# Tracer Capacité du canal vs SNR
plt.figure(figsize=(10,6))
plt.plot(snr_values, capacity_autoencoder_list, 'g-d', label='Capacité du canal')
plt.xlabel('SNR (dB)')
plt.ylabel('Capacité (bits/s/Hz)')
plt.title('Capacité du canal en fonction du SNR')
plt.grid(True)
plt.legend()
plt.show()

# Tracer Latence de transmission vs SNR
plt.figure(figsize=(10,6))
plt.plot(snr_values, latency_autoencoder_list, 'm', label='Latence de transmission')
plt.xlabel('SNR (dB)')
plt.ylabel('Temps (s)')
plt.title('Latence de transmission en fonction du SNR')
plt.grid(True)
plt.legend()
plt.show()