import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math

import matplotlib.pyplot as plt

from IPython.utils import io
from IPython import display

from channel import channel
from models import Encoder, Decoder
from utils import MemoryMessages, count_errors

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train = True

def train_autoencoder(m, n, snr_db, chann_type, batch_size, n_epochs, lr, clipping, plot, stop_value):

    
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
            data_channel = channel(encoded_data, snr_db, chann_type=chann_type)
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
                print("Le modele a converge.")
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

def evaluate_autoencoder(encoder, decoder, m, n, k, snr_db, chann_type, n_samples=10000):
    """
    Évalue l'autoencodeur en mesurant le BER sur un grand nombre d'exemples.
    """
    encoder.eval()
    decoder.eval()
    errors = 0

    for _ in range(n_samples):
        # Générer un message aléatoire
        message = torch.randint(0, m, (1,)).to(device)

        # Encoder le message
        encoded = encoder(message)

        # Appliquer le canal en appelant directement la fonction `channel()`
        received = channel(encoded, n, k, snr_db, chann_type)

        # Decoder le message
        decoded = decoder(received)
        predicted = torch.argmax(decoded, dim=1)

        # Compter les erreurs
        errors += (predicted != message).sum().item()

    ber = errors / n_samples
    print(f"BER (SNR={snr_db} dB, Canal={chann_type}): {ber:.6f}")
    return ber

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
    encoder, decoder, errors = train_autoencoder(m=16, n=7,snr_db=7 ,chann_type="Rician", batch_size=64, n_epochs=10000, lr=0.001,
                                clipping=0.5, plot=10, stop_value=0.005)    
    
    plot_training_loss(errors)  # Affichage de l'évolution de la perte