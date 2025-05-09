# DeepComm-Autoencoder-Realistic-Channels

## 📌 Project Overview
This project explores the use of **autoencoders** in digital communication systems, focusing on **realistic wireless channels**.  
Instead of assuming ideal AWGN channels, we integrate **fading models** (Rayleigh, Rician) and incorporate **Channel State Information (CSI)** to enhance system adaptability.  

## 🎯 Objectives
- Implement an **autoencoder-based communication system** optimized for **realistic channels**.
- Study the impact of **CSI feedback** on performance.
- Design a **joint transmitter-receiver model** using **deep learning**.
- Integrate **modulation constraints** to ensure practical feasibility.

## 🏗️ Project Structure
```
📂 DeepComm-Autoencoder-Realistic-Channels 
│── 📄 README.md # Project documentation 
│── 📄 requirements.txt # Dependencies
│── 📄 channel.py # Simulates channels
│── 📄 com_System.py # Simulates an QPSK communication system
│── 📄 debug_test.py # Debugging and testing script
│── 📄 evaluate.py # Model evaluation and performance analysis
│── 📄 models.py # Neural network architectures 
│── 📄 modulation.py # Modulation constraints and utilities
│── 📄 train.py # Main training script
│── 📄 utils.py # Utility functions
│── 📂 saved_models # Folder to store trained models
└── 📂 plots # Generated plots
```


## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/senhajibooutayna/DeepComm-Autoencoder-Realistic-Channels
cd DeepComm-Autoencoder-Realistic-Channels
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run a Basic Test
```bash
python train.py
```
