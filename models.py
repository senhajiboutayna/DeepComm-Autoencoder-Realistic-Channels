import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Pytorch device: {device}")

class Encoder(nn.Module):

    def __init__(self, m, n, dim=512):
        super(Encoder, self).__init__()
        
        self.n = n

        self.linear_M = nn.Sequential(
            nn.Embedding(num_embeddings=m, embedding_dim=m),
            nn.ReLU(),
        )

        self.linear_N = nn.Sequential(
            nn.Linear(in_features=m, out_features=n),
        )
               
        self.normalization = nn.BatchNorm1d(num_features=n)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
    def forward(self,x):
        x = self.linear_M(x)
        x = self.linear_N(x.squeeze())
        y = self.normalization(x)
        y = y / torch.sqrt(torch.mean(y ** 2))    # Power normalization
        """
        Currently, BatchNorm1d is used, but in a real case, the signal must respect a power constraint.
        Idea: Force an average power of 1 with explicit normalization
        This will ensure that the transmitter does not exceed the permitted power.
        """
        return y


class Decoder(nn.Module):

    def __init__(self, m, n):

        super(Decoder, self).__init__()

        self.n = n

        self.linear_relu = nn.Sequential(
            nn.Linear(in_features=n, out_features=m),
            nn.ReLU(),
        )
        
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=m, out_features=m),
            nn.LogSoftmax(dim=1),
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, y):

        y = y.view(-1, self.n)  

        # Decoding phase
        y = self.linear_relu(y)
        y = self.linear_out(y)
        
        return y
    

class FeedbackCorrection(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, robust=False):
        super(FeedbackCorrection, self).__init__()

        if robust:
            input_dim = input_dim * 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):

        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder

class RobustEncoder(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.m = m
        self.n = n

        # Embedding layer
        self.embed = nn.Embedding(m, m)
        
        # Modifier les conv1d pour gérer correctement les dimensions
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        self.fc = nn.Linear(32, n)
        self.norm = nn.BatchNorm1d(n)
        
    def forward(self, x):
        # x shape: [batch_size, 1]
        if x.dim() > 1:
            x = self.embed(x.squeeze(1))
        else:
            x = self.embed(x)
        x = x.unsqueeze(1)  # [batch_size, 1, embedding_dim] pour conv1d
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, features] pour attention
        x, _ = self.attention(x, x, x)
        x = x.mean(dim=0)  # Pooling temporel [batch_size, features]
        x = self.fc(x)
        x = self.norm(x)
        return x / torch.norm(x, p=2, dim=1, keepdim=True) * np.sqrt(x.size(1))

class RobustDecoder(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.lstm = nn.LSTM(n, 64, num_layers=2, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.fc1 = nn.Linear(128, m)
        self.fc2 = nn.Linear(m, m)
        
    def forward(self, y):
        y = y.unsqueeze(0).repeat(4, 1, 1)  # Créer une séquence
        y, _ = self.lstm(y)
        y, _ = self.attention(y, y, y)
        y = y.mean(dim=0)  # Pooling temporel
        y = F.relu(self.fc1(y))
        return F.log_softmax(self.fc2(y), dim=1)
    

class Transmitter(nn.Module):
    def __init__(self, m=16, n=7, hidden=64, use_csi=False):
        super().__init__()
        self.use_csi = use_csi
        # entrée : one-hot de taille M, éventuellement concaténée avec CSI (2 réels)
        dim_in = m + (2 if use_csi else 0)
        self.fc1 = nn.Linear(dim_in, hidden)
        self.fc2 = nn.Linear(hidden, 2*n)  # 2*n => n symboles complexes
    def forward(self, x, h=None):
        # x est one-hot (batch, M), h est (batch,2) pour CSI (re,im)
        if self.use_csi and h is not None:
            x = torch.cat([x, h], dim=1)
        z = F.relu(self.fc1(x))
        z = self.fc2(z)  # shape (batch, 2*n)
        # Normaliser la puissance moyenne à 1 (soft constraint)
        power = torch.mean(z.pow(2).sum(dim=1))
        z = z / torch.sqrt(power + 1e-9)
        return z  # vecteur réel (2*n) à transmettre

class Receiver(nn.Module):
    def __init__(self, n=7, m=16, hidden=64, use_csi=False, use_ML=False):
        super().__init__()
        self.use_csi = use_csi
        self.use_ML = use_ML
        # entrée : 2*n valeurs réelles, éventuellement concaténation de CSI
        dim_in = 2*n + (2 if use_csi else 0)
        self.fc_reduce = nn.Linear(28, 16)
        self.fc_reduce2 = nn.Linear(30, 16)
        self.fc1 = nn.Linear(dim_in, hidden)
        self.fc2 = nn.Linear(hidden, m)
    def forward(self, y, h=None):
    
        # y shape (batch, 2*n), h shape (batch,2)
        if self.use_csi and h is not None:
            y = torch.cat([y, h], dim=1)
            y = F.relu(self.fc_reduce(y))
        
        if self.use_ML :
            y = torch.cat([y, h], dim=1)
            y = F.relu(self.fc_reduce2(y))
    
        y = F.relu(self.fc1(y))
        out = self.fc2(y)  # logits (batch, M)
        return F.log_softmax(out, dim=1)
    
