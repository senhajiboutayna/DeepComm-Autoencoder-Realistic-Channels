import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pyldpc

def generate_ldpc_code(n, k):
    """
    Generates an LDPC matrix (H) and a generator matrix (G).
    
    n: Total number of bits (code length).
    k : Number of information bits.
    
    Returns H (parity check matrix) and G (generator matrix).
    """
    d_v = 3  # Degree of control bits
    d_c = 6  # Degree of parity bits

    # Correction: Adapt n to be a multiple of d_c
    n = (n // d_c + 1) * d_c if n % d_c != 0 else n

    H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    return H, G

def apply_doppler_effect(symbols, doppler_freq, sf):
    """
    Applies Doppler phase noise to an OFDM signal to simulate a mobile channel.
    
    doppler_freq: Doppler frequency (Hz).
    sf : Sampling frequency.
    """
    if doppler_freq == 0:
        return symbols  # No Doppler
    n = symbols.shape[1]
    t = np.arange(n) / sf
    phase_shift = np.exp(1j * 2 * np.pi * doppler_freq * t)
    return symbols * phase_shift


def evaluate_ofdm(snr_db, chann_type="AWGN", num_symbols=10, M=16, K=64, CP=16, K_rician=3, doppler_freq=0, use_ldpc=False):
    """
    Simulates an OFDM system with channel and adaptive modulation.
    
    Parameters:
        snr_db (float): SNR in dB.
        chann_type (str): Channel type (“AWGN”, “Rayleigh”, “Rician”).
        num_symbols (int): Number of OFDM symbols.
        M (int): QAM modulation order.
        K (int): Number of OFDM subcarriers.
        CP (int): Cyclic prefix length, (25% of the block k) -> 16
        K_rician (float): Rician channel K factor.
        doppler_freq (float): Doppler frequency to simulate a mobile channel.
        use_ldpc (bool): Use LDPC coding.
    
    Return:
        ber (float): Bit error rate.
    """

    # Adapting the modulation scheme to the SNR
    if snr_db < 5:
        M = 4  # QPSK
    elif snr_db < 11:
        M = 16  # 16-QAM
    else:
        M = 64  # 64-QAM
    
    # Modulation Parameters
    bits_per_symbol = int(np.log2(M))
    num_bits = num_symbols * K * bits_per_symbol
    bits = np.random.randint(0, 2, num_bits)   # Generate random bits
    print ("Bits count: ", len(bits))
    print ("Mean of bits (should be around 0.5): ", np.mean(bits))

    # LDPC coding (if enabled)   
    if use_ldpc:
        H, G = generate_ldpc_code(n=num_bits, k=num_bits // 2)
        k_ldpc = G.shape[1]  # Number of information bits expected by G
        bits_input = bits[:k_ldpc]  # Adjust bit size
        bits = pyldpc.encode(G, bits_input, snr_db)
        bits = (bits > 0.5).astype(int)  # Convert to binary (0 or 1)

    # QAM Modulation
    def qam_mod(bits, M):
        """QAM modulation with Gray mapping"""
        k = int(np.log2(M))
        # Ajustement pour que bits soit un multiple de k
        extra_bits = bits.size % k
        if extra_bits != 0:
            print(f"Avertissement: Tronquage de {extra_bits} bits pour correspondre à QAM-{M}")
            bits = bits[:bits.size - extra_bits]

        bits = bits.reshape(-1, k)
        I_levels = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
        Q_levels = np.arange(-np.sqrt(M) + 1, np.sqrt(M), 2)
        constellation = np.array([i + 1j * q for i in I_levels for q in Q_levels])
        """
        # Vérifier que bits contient uniquement 0 et 1
        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("Erreur: les bits fournis à qam_mod() ne sont pas binaires !")
        """
        indices = np.dot(bits, 2**np.arange(k)[::-1]).astype(int)

        # Check that the indices are in  [0, M-1]
        if np.any(indices < 0) or np.any(indices >= M):
            raise ValueError(f"Erreur: Indices hors limites. Min: {indices.min()}, Max: {indices.max()}, M={M}")

        symbols = constellation[indices]
        symbols /= np.sqrt((np.mean(np.abs(symbols) ** 2)))

        print(f"Indices QAM min: {indices.min()}, max: {indices.max()}, M={M}")
    
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
    if chann_type == "AWGN":
        h = np.ones(K)

    elif chann_type == "Rayleigh":
        h = (np.random.randn(K) + 1j * np.random.randn(K)) / np.sqrt(2)
    
    elif chann_type == "Rician":
        h_los = np.ones(K)
        h_nlos = (np.random.randn(K) + 1j * np.random.randn(K)) / np.sqrt(2)
        h = np.sqrt(K_rician / (K_rician + 1)) * h_los + np.sqrt(1 / (K_rician + 1)) * h_nlos
    else:
        raise ValueError(f"Type de canal non supporté: {chann_type}")

    # QAM Mapping
    symbols = qam_mod(bits, M)
    ##expected_size = num_symbols * len(dataCarriers)
    ##symbols = symbols[:expected_size]  # Tronquer si nécessaire
    expected_size = num_symbols * K  # Taille correcte attendue

    if symbols.size < expected_size:
        raise ValueError(f"Erreur: Pas assez de symboles ({symbols.size}) pour remplir ({num_symbols}, {K})")
    elif symbols.size > expected_size:
        print(f"Avertissement: Tronquage de {symbols.size} → {expected_size} pour correspondre à ({num_symbols}, {K})")
        symbols = symbols[:expected_size]  # Truncate the excess

    symbols = symbols.reshape(num_symbols, K)

    # IFFT Transformation (Time Domain Conversion)
    ofdm_symbols = np.fft.ifft(symbols, axis=1)

    # Adding Cyclic Prefix
    cp = ofdm_symbols[:, -CP:]
    ofdm_tx = np.hstack([cp, ofdm_symbols])

    # Passage through the channel
    h = h[np.newaxis, :]
    h = np.tile(h, (num_symbols, 1))
    ofdm_rx = ofdm_tx[:, CP:] * h 

    # Adding AWGN Noise
    snr_linear = 10 ** (snr_db / 10)
    noise_power = np.mean(np.abs(ofdm_tx)**2) / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*ofdm_tx.shape) + 1j * np.random.randn(*ofdm_tx.shape))
    ofdm_rx = ofdm_tx + noise

    signal_power = np.var(ofdm_tx)
    noise_power = np.var(noise)
    snr_measured = 10 * np.log10(signal_power / noise_power)
    print(f"SNR théorique: {snr_db} dB, SNR mesuré: {snr_measured:.2f} dB")


    # Add phase noise (Doppler)
    ofdm_rx = apply_doppler_effect(ofdm_rx, doppler_freq, sf=K)

    # Deletion of cyclic prefix
    ofdm_rx = ofdm_rx[:, CP:]

    # FFT reception
    symbols_rx = (np.fft.fft(ofdm_rx, axis=1) * h.conj()) / (np.abs(h)**2 + 1e-6) # Equalization (Division by the channel response) - Égalisation

    # Demodulation
    bits_rx = qam_demod(symbols_rx.flatten(), M)

    # LDPC decoding (if enabled).
    if use_ldpc:
        bits_rx = pyldpc.decode(H, bits_rx, snr_db, maxiter=100) 

    # Calculation of BER
    bits = bits[:len(bits_rx)]  # Tronquer bits à la même longueur que bits_rx
    ber = np.mean(bits != bits_rx)
    print(f"BER (OFDM + {M}-QAM + {chann_type.upper()}, SNR={snr_db} dB) : {ber:.6f}")
    
    # Display of received constellation
    plt.figure(figsize=(6,6))
    plt.scatter(symbols.flatten().real, symbols.flatten().imag, marker='x', color='b', alpha=0.5, label="Émis")
    plt.scatter(symbols_rx.flatten().real, symbols_rx.flatten().imag, marker='o', color='r', alpha=0.3, label="Reçu")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.legend()
    plt.title(f"Constellation Reçue ({M}-QAM, {chann_type} Channel, SNR={snr_db} dB)")
    plt.grid()
    plt.show()

    return ber



