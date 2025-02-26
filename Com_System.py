import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def evaluate_ofdm(snr_db, chann_type="AWGN", num_symbols=100, M=16, K=64, CP=16, K_rician=3, use_ldpc=True):
    """
    Évalue le système OFDM pour un SNR donné et retourne le BER.
    
    Paramètres:
        snr_db (float): SNR en dB.
        chann_type (str): Type de canal ("AWGN", "Rayleigh", "Rician").
        num_symbols (int): Nombre de symboles OFDM à générer.
        M (int): Ordre de la modulation QAM.
        K (int): Nombre de sous-porteuses OFDM.
        CP (int): Longueur du préfixe cyclique.
        K_rician (float): Facteur de Rician pour le canal Rician.
        use_ldpc (bool): Active/Désactive le codage correcteur d'erreur (LDPC).
    
    Retour:
        ber (float): Taux d'erreur binaire (BER) calculé.
    """

    # Sélection dynamique du MCS (Modulation and Coding Scheme)
    if snr_db < 5:
        M = 4  # QPSK
    elif snr_db < 11:
        M = 16  # 16-QAM
    else:
        M = 64  # 64-QAM

    # OFDM Parameters
    K = 64  # Number of OFDM subcarriers
    CP = K // 4  # Length of the cyclic prefix (25% of the block) -> 16
    P = CP // 2  # Number of pilot carriers per OFDM block
    pilotValue = 3 + 3j  # Known value transmitted by pilots
    snr_db = 25  # SNR in dB
    num_symbols = 100  # Number of OFDM symbols
    K_rician = 3  # Rician factor for the channel

    # Define the indices of the subcarriers
    allCarriers = np.arange(K)
    pilotCarriers = allCarriers[::K // P]
    pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
    P = len(pilotCarriers)
    dataCarriers = np.delete(allCarriers, pilotCarriers)
    print ("allCarriers:   %s" % allCarriers)
    print ("pilotCarriers: %s" % pilotCarriers)
    print ("dataCarriers:  %s" % dataCarriers)

    # Visualization of subcarriers
    plt.figure(figsize=(8,4))
    plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='Pilots')
    plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='Data')
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Magnitude")
    plt.title("OFDM Subcarrier Allocation")
    plt.legend()
    plt.grid()
    #plt.show()

    # QAM Modulation
    def qam_mod(bits, M):
        """QAM modulation with Gray mapping"""
        k = int(np.log2(M))
        bits = bits.reshape(-1, k)
        I_levels = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
        Q_levels = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
        constellation = np.array([i + 1j * q for i in I_levels for q in Q_levels])
        indices = np.dot(bits, 2**np.arange(k)[::-1])
        symbols = constellation[indices]
        symbols /= np.sqrt((np.mean(np.abs(constellation) ** 2)))
    
        # Visualization of the QAM constellation
        plt.figure(figsize=(6,6))
        plt.scatter(symbols.real, symbols.imag, marker='x', color='b')
        plt.xlabel("In-phase")
        plt.ylabel("Quadrature")
        plt.title(f"{M}-QAM Constellation")
        plt.grid()
        #plt.show()
    
        return symbols

    # QAM Demodulation
    def qam_demod(symbols_rx, M):
        """QAM demodulation using ML decision"""
        k = int(np.log2(M))
        I_levels = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
        Q_levels = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
        constellation = np.array([i + 1j * q for i in I_levels for q in Q_levels])
        indices = np.argmin(np.abs(symbols_rx[:, None] - constellation[None, :]), axis=1)
        bits_rx = np.array([[int(b) for b in np.binary_repr(i, width=k)] for i in indices])
        return bits_rx.flatten()

    # Channel Generation Functions
    def awgn_channel(N_subcarriers):
        """Generates an AWGN channel"""
        return np.ones(N_subcarriers)

    def rayleigh_channel(N_subcarriers):
        """Generates a Rayleigh fading channel"""
        return (np.random.randn(N_subcarriers) + 1j * np.random.randn(N_subcarriers)) / np.sqrt(2)

    def rician_channel(N_subcarriers, K):
        """Generates a Rician fading channel with factor K"""
        h_los = np.ones(N_subcarriers)
        h_nlos = (np.random.randn(N_subcarriers) + 1j * np.random.randn(N_subcarriers)) / np.sqrt(2)
        h = np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos
        return h

    # Modulation Parameters
    bits_per_symbol = int(np.log2(M))
    payloadBits_per_OFDM = len(dataCarriers) * bits_per_symbol
    num_total_bits = num_symbols * payloadBits_per_OFDM
    bits = np.random.randint(0, 2, size=num_total_bits)
    print ("Bits count: ", len(bits))
    print ("First 20 bits: ", bits[:20])
    print ("Mean of bits (should be around 0.5): ", np.mean(bits))

    # Codage correcteur d'erreur (LDPC)
    if use_ldpc:
        encoded_bits = np.repeat(bits, 2)  # Simplification d'un codage LDPC (x2 bits)
        encoded_bits = encoded_bits[:num_total_bits]  # S'assurer que la taille reste correcte
    else:
        encoded_bits = bits

    # QAM Mapping
    symbols = qam_mod(encoded_bits, M)
    expected_size = num_symbols * len(dataCarriers)
    symbols = symbols[:expected_size]  # Tronquer si nécessaire
    symbols = symbols.reshape(num_symbols, len(dataCarriers))

    # Create OFDM Frame
    ofdm_frame = np.zeros((num_symbols, K), dtype=complex)
    ofdm_frame[:, dataCarriers] = symbols
    ofdm_frame[:, pilotCarriers] = pilotValue

    # IFFT Transformation (Time Domain Conversion)
    ofdm_symbols = np.fft.ifft(ofdm_frame, axis=1)

    # Visualization of OFDM symbols
    plt.figure(figsize=(8,4))
    plt.plot(abs(ofdm_symbols[0]), label='OFDM Symbol Magnitude')
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Magnitude")
    plt.title("OFDM Symbol Representation")
    plt.grid()
    plt.legend()
    #plt.show()

    # Adding Cyclic Prefix
    cp = ofdm_symbols[:, -CP:]
    ofdm_tx = np.hstack([cp, ofdm_symbols])

    # Selection of the channel
    chann_type = "AWGN"  
    if chann_type == "AWGN":
        h = awgn_channel(K)  
    elif chann_type == "Rayleigh":
        h = rayleigh_channel(K)  
    elif chann_type == "Rician":
        h = rician_channel(K, K_rician)  
    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}") 

    # Passage par le canal
    h = h[np.newaxis, :]
    h = np.tile(h, (num_symbols, 1))
    ofdm_rx = ofdm_tx[:, CP:] * h 

    # Adding AWGN Noise
    snr_linear = 10 ** (snr_db / 10)
    noise_power = np.mean(np.abs(ofdm_tx)**2) / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*ofdm_tx.shape) + 1j * np.random.randn(*ofdm_tx.shape))
    ofdm_rx = ofdm_tx + noise

    # Visualization of noisy signal
    plt.figure(figsize=(8,4))
    plt.plot(abs(ofdm_rx[0]), label='Received OFDM Symbol')
    plt.xlabel("Time Sample Index")
    plt.ylabel("Magnitude")
    plt.title("Received OFDM Symbol with Noise")
    plt.grid()
    plt.legend()
    #plt.show()

    # Deletion of cyclic prefix
    ofdm_rx = ofdm_rx[:, CP:]

    # FFT reception
    symbols_rx = np.fft.fft(ofdm_rx, axis=1) / h  # Equalization (Division by the channel response) - Égalisation

    # Extraction of useful data (without pilots)
    symbols_rx_data = symbols_rx[:, dataCarriers]

    # Demodulation
    bits_rx = qam_demod(symbols_rx_data.flatten(), M)

    # Décodage LDPC (si activé)
    if use_ldpc:
        bits_rx = bits_rx[::2]  # Suppression du doublage de bits

    # Calculation of BER
    bits = bits[:bits_rx.shape[0]]  # Tronquer bits à la même longueur que bits_rx
    ber = np.mean(bits != bits_rx)
    print(f"BER (OFDM + {M}-QAM + {chann_type.upper()}, SNR={snr_db} dB) : {ber:.6f}")

    # Display of received constellation
    plt.figure(figsize=(6,6))
    plt.scatter(symbols_rx_data.flatten().real, symbols_rx_data.flatten().imag, marker='o', color='r', alpha=0.3)
    plt.scatter(symbols.real, symbols.imag, marker='x', color='b', alpha=0.5)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(f"Constellation Reçue ({M}-QAM, {chann_type} Channel)")
    plt.grid()
    #plt.show()

    return ber

"""
# Test du système OFDM
snr_range = np.arange(-4, 10, 1)
ber_values_awgn = [evaluate_ofdm(snr, "AWGN", use_ldpc=True) for snr in snr_range]
ber_values_rayleigh = [evaluate_ofdm(snr, "Rayleigh", use_ldpc=True) for snr in snr_range]
ber_values_rician = [evaluate_ofdm(snr, "Rician", use_ldpc=True) for snr in snr_range]

# Affichage du BER vs SNR
plt.figure(figsize=(8,5))
plt.semilogy(snr_range, ber_values_awgn, 'r-o', label="OFDM AWGN")
plt.semilogy(snr_range, ber_values_rayleigh, 'g-s', label="OFDM Rayleigh")
plt.semilogy(snr_range, ber_values_rician, 'b-d', label="OFDM Rician")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.grid()
plt.title("Performance OFDM avec Codage LDPC")
plt.show()
"""