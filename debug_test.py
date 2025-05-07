from com_System import qpsk_communication
from channel import channel

import numpy as np
import torch
import matplotlib.pyplot as plt


def test_rayleigh_with_qpsk(snr_db_values, n_bits):
    """Test QPSK transmission through Rayleigh channel without ML"""
    ber_results = []
    
    for snr_db in snr_db_values:
        ber = qpsk_communication(snr_db,num_bits=n_bits, channel_type="Rayleigh")
        ber_results.append(ber.item())
    
    # Plot
    plt.semilogy(snr_db_values, ber_results, 'o-', label="Rayleigh")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Performance QPSK sur canal Rayleigh")
    plt.grid(True)
    plt.legend()
    plt.show()


snr_db_values = np.arange(-5, 20, 2)
n_bits = 50000

#test_rayleigh_with_qpsk(snr_db_values, n_bits)

def test_channel_normalization():
    n_samples = 10000
    x = torch.ones(n_samples)  # Signal unitaire
    _, _, _, _, h, _ = channel(x, snr_db=10, chann_type="Rayleigh")
    
    h_np = h.numpy()
    avg_power = np.mean(np.abs(h_np)**2)
    print(f"Puissance moyenne du canal (doit être ~1.0) : {avg_power:.4f}")
    
    plt.hist(np.abs(h_np), bins=50, density=True)
    plt.xlabel('|h|')
    plt.ylabel('Densité')
    plt.title('Distribution des coefficients du canal Rayleigh')
    plt.grid()
    plt.show()

test_channel_normalization()

