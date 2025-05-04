import train  # Assure-toi que train.py est dans le même dossier
import torch

def run_test_awgn_vs_rayleigh():
    print("\n[TEST 1] AWGN vs Rayleigh")
    for ch in ["AWGN", "Rayleigh"]:
        print(f"→ Channel Type: {ch}")
        train.train_autoencoder(
            chann_type=ch,
            m=4,
            n=2,
            snr_db=10,
            num_epochs=10,
            sigma_CSI=0.0,
            batch_size=128,
            learning_rate=1e-3,
            visualize=False,
            feedback=False,
        )

def run_test_power_norm():
    print("\n[TEST 2] Power Normalization")
    x = torch.randn(128, 2)
    norm = torch.norm(x, dim=-1)
    print(f"→ Sample Power Norms: {norm[:10]}")
    print(f"→ Mean Power: {torch.mean(norm)}")

def run_test_sigma_CSI_variation():
    print("\n[TEST 3] Variation de sigma_CSI")
    for sigma in [0.0, 0.5, 1.0, 2.0]:
        print(f"→ sigma_CSI = {sigma}")
        train.train_autoencoder(
            chann_type="Rayleigh",
            m=4,
            n=2,
            snr_db=10,
            num_epochs=10,
            sigma_CSI=sigma,
            batch_size=128,
            learning_rate=1e-3,
            visualize=False,
            feedback=False,
        )

def run_test_feedback_vs_no():
    print("\n[TEST 4] Feedback ON/OFF")
    for fb in [False, True]:
        print(f"→ Feedback: {'ON' if fb else 'OFF'}")
        if fb:
            train.train_autoencoder_with_feedback(
                m=4,
                n=2,
                snr_db=10,
                num_epochs=10,
                sigma_CSI=0.5,
                batch_size=128,
                learning_rate=1e-3,
                visualize=False,
                use_ml_feedback=False,
                compression_level=1.0,
                snr_feedback=15,
                delay=0
            )
        else:
            train.train_autoencoder(
                chann_type="Rayleigh",
                m=4,
                n=2,
                snr_db=10,
                num_epochs=10,
                sigma_CSI=0.5,
                batch_size=128,
                learning_rate=1e-3,
                visualize=False,
                feedback=False,
            )

def run_test_ml_feedback_vs_no():
    print("\n[TEST 5] ML Feedback ON/OFF")
    for ml in [False, True]:
        print(f"→ ML Feedback: {'ON' if ml else 'OFF'}")
        train.train_autoencoder_with_feedback(
            m=4,
            n=2,
            snr_db=10,
            num_epochs=10,
            sigma_CSI=0.5,
            batch_size=128,
            learning_rate=1e-3,
            visualize=False,
            use_ml_feedback=ml,
            compression_level=1.0,
            snr_feedback=15,
            delay=0
        )

if __name__ == "__main__":
    print("=== DEBUGGING RUNNER ===")
    run_test_awgn_vs_rayleigh()
    run_test_power_norm()
    run_test_sigma_CSI_variation()
    run_test_feedback_vs_no()
    run_test_ml_feedback_vs_no()
