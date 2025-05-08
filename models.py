import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Pytorch device: {device}")

class Encoder(nn.Module):

    def __init__(self, m, n, dim=512):
        super(Encoder, self).__init__()
        
        self.m = m
        self.n = n

        self.embed = nn.Sequential(
            nn.Embedding(num_embeddings=m, embedding_dim=m),
            nn.ReLU(),
        )

        # Branche convolutionnelle
        self.conv_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Mécanisme d'attention
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)


        self.linear_branch = nn.Sequential(
            nn.Linear(in_features=m, out_features=n),
            nn.ReLU(),
        )

        # Fusion des branches
        self.fusion = nn.Linear(n + 32, n)
               
        self.batch_norm = nn.BatchNorm1d(num_features=n)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)
        
    def forward(self,x):
        # Gestion des dimensions pour assurer la compatibilité
        if x.dim() > 1:
            x = x.squeeze(1)
        
        # Branche commune d'embedding
        embedded = self.embed(x)
        
        # Branche linéaire
        linear_output = self.linear_branch(embedded)
        
        # Branche convolutionnelle
        conv_input = embedded.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        conv_output = self.conv_branch(conv_input)
        
        # Application de l'attention
        conv_output = conv_output.permute(2, 0, 1)  # [seq_len, batch_size, features]
        attn_output, _ = self.attention(conv_output, conv_output, conv_output)
        attn_output = attn_output.mean(dim=0)  # Pooling temporel [batch_size, features]
        
        # Fusion des branches
        combined = torch.cat([linear_output, attn_output], dim=1)
        output = self.fusion(combined)
        
        # Normalisation
        output = self.batch_norm(output)
        
        # Normalisation de puissance
        y = output / torch.norm(output, p=2, dim=1, keepdim=True) * np.sqrt(output.size(1))

        return y


class Decoder(nn.Module):

    def __init__(self, m, n):

        super(Decoder, self).__init__()

        self.n = n

        # LSTM bidirectionnel
        self.lstm = nn.LSTM(n, 64, num_layers=2, bidirectional=True)
        
        # Mécanisme d'attention
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)

        self.linear_relu = nn.Sequential(
            nn.Linear(in_features=128, out_features=m),
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

         # Transformation pour LSTM [sequence_length, batch_size, input_size]
        y = y.unsqueeze(0).repeat(4, 1, 1)  # Créer une séquence temporelle artificielle
        
        # Passage dans le LSTM
        lstm_out, _ = self.lstm(y)
        
        # Application de l'attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Pooling temporel
        attn_out = attn_out.mean(dim=0)  # [batch_size, features]
        
        # Décodage final avec les couches linéaires
        out = self.linear_relu(attn_out)
        y = self.linear_out(out) 

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