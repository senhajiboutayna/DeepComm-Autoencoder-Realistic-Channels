from com_System import qpsk_communication
from channel import channel, feedback_csi, generate_correlated_rayleigh

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

test_rayleigh_with_qpsk(snr_db_values, n_bits)

def test_channel_normalization():
    n_samples = 10000
    x = torch.ones(n_samples)  # Signal unitaire
    _, _, _, _, h, _ = channel(x, snr_db=7, chann_type="Rayleigh")
    
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

def test_csi_consistency():
    x = torch.tensor([1.0, -1.0, 0.5, -0.5])  # Exemple de signal
    snr_db = 7
    sigma_CSI = 0.5  # Bruit sur le CSI
    
    
    # Passage dans le canal
    x_channel, _, x_received, x_received_CSI, h, h_hat = channel(
        x, snr_db, chann_type="Rayleigh", sigma_CSI=sigma_CSI
    )
    
    
    print("=== Vérification CSI ===")
    print(f"Vrai canal (h)      : {h}")
    print(f"CSI estimé (h_hat)  : {h_hat}")
    print(f"Erreur moyenne      : {torch.mean(torch.abs(h - h_hat)):.4f}")
    
    # Plot des distributions
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.hist(h.numpy(), bins=20, alpha=0.7, label='Vrai canal (h)')
    plt.hist(h_hat.numpy(), bins=20, alpha=0.7, label='CSI estimé (h_hat)')
    plt.xlabel('Valeur')
    plt.ylabel('Densité')
    plt.legend()
    plt.grid()
    
    plt.subplot(122)
    plt.scatter(h.numpy(), h_hat.numpy(), alpha=0.6)
    plt.plot([0, 2], [0, 2], 'r--', label='CSI parfait')
    plt.xlabel('h (vrai)')
    plt.ylabel('h_hat (estimé)')
    plt.legend()
    plt.grid()
    plt.show()

test_csi_consistency()

def test_block_fading():
    n_symbols = 10000
    x = torch.ones(n_symbols)
    _, _, _, _, h, _ = channel(x, snr_db=7, chann_type="Rayleigh")
    
    # Plot des coefficients sur les 50 premiers symboles
    plt.plot(h[:50].numpy(), 'o-', label='Coefficients du canal')
    plt.xlabel('Symbole')
    plt.ylabel('Valeur de h')
    plt.title('Évolution du canal Rayleigh')
    plt.grid()
    plt.legend()
    plt.show()

test_block_fading()

def test_feedback():
    h = torch.randn(10)  # Exemple de canal
    h_hat = feedback_csi(h, snr_feedback=7, compression_level=4, delay=2)
    
    plt.scatter(h.numpy(), h_hat.numpy(), alpha=0.6)
    plt.plot([-3, 3], [-3, 3], 'r--', label='CSI parfait')
    plt.xlabel('h (vrai)')
    plt.ylabel('h_hat (feedback)')
    plt.legend()
    plt.grid()
    plt.show()

test_feedback()

def plot_fading_correlation():
    h_fast = torch.sqrt(torch.randn(1000)**2 + torch.randn(1000)**2) / np.sqrt(2)  # Fast fading
    h_slow = generate_correlated_rayleigh(1000, rho=0.99)  # Slow fading
    
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(h_fast[:200].numpy(), 'r-', label="Fast Fading")
    plt.title("Fast Fading (indépendant)")
    plt.grid()
    
    plt.subplot(122)
    plt.plot(h_slow[:200].numpy(), 'b-', label=f"Slow Fading (ρ=0.99)")
    plt.title("Correlated Fading (AR1)")
    plt.grid()
    plt.show()

plot_fading_correlation()