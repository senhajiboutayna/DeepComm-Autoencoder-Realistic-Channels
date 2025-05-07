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
        y = y / torch.norm(y, p=2, dim=1, keepdim=True) * np.sqrt(x.size(1))    # Power normalization
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
    def __init__(self, input_dim, hidden_dim=128):
        super(FeedbackCorrection, self).__init__()

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