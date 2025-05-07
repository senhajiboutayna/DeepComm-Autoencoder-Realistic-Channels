import torch
import torch.nn as nn
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

class TemporalEncoder(nn.Module):
    """
    Encodeur avec capacité d'adaptation temporelle pour gérer le fast fading.
    """
    def __init__(self, m, n, hidden_dim=512, seq_len=10):
        super(TemporalEncoder, self).__init__()

        self.n = n
        self.seq_len = seq_len  # Longueur de la séquence temporelle à considérer

        # Embedding initial des symboles
        self.embedding = nn.Embedding(num_embeddings=m, embedding_dim=m)

        # Couche LSTM pour capturer les dépendances temporelles
        self.lstm = nn.LSTM(
            input_size=m,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # Mécanisme d'attention pour se concentrer sur les parties importantes du signal
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim*2,  # *2 car bidirectionnel
            num_heads=4,
            dropout=0.1
        )

        # Projection finale vers l'espace du canal
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n)
        )
        
        # Normalisation de la puissance
        self.normalization = nn.BatchNorm1d(num_features=n)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input x shape: [batch_size]
        
        # Create sequence by expanding to seq_len
        # This step might need to be adapted depending on your actual input format
        if x.dim() == 1:
            # If x is just batch_size, we expand it to create a sequence
            x = x.unsqueeze(1).expand(-1, self.seq_len)
        
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)
        
        # LSTM expects [batch_size, seq_len, input_size]
        # No need for additional unsqueeze
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]

        # Mécanisme d'attention
        # Transpose for attention: [seq_len, batch_size, hidden_dim*2]
        lstm_out = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # [batch_size, seq_len, hidden_dim*2]

        # On ne garde que le dernier état (ou on peut faire une moyenne)
        feature = attn_out[:, -1, :]  # [batch_size, hidden_dim*2]

        # Projection finale
        y = self.projection(feature)  # [batch_size, n]

        # Normalisation de la puissance
        y = self.normalization(y)
        y = y / torch.norm(y, p=2, dim=1, keepdim=True) * np.sqrt(y.size(1))

        return y
    

class TemporalDecoder(nn.Module):
    """
    Décodeur avec capacité d'adaptation temporelle pour gérer le fast fading.
    """
    def __init__(self, m, n, hidden_dim=512):
        super(TemporalDecoder, self).__init__()
        
        self.n = n
        
        # Module d'estimation du canal pour aider à compenser les variations
        self.channel_estimator = nn.Sequential(
            nn.Linear(n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n),
            nn.Tanh()
        )
        
        # Décodeur récurrent
        self.gru = nn.GRU(
            input_size=n,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Projection finale vers l'espace des symboles
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, m),
            nn.LogSoftmax(dim=1)
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, y):
        # y: [batch_size, n] (la sortie du canal)
        
        # Reshape si nécessaire
        y = y.view(-1, self.n)
        
        # Estimation et compensation du canal (optionnel)
        channel_estimate = self.channel_estimator(y)
        y_compensated = y + channel_estimate  # Compensation adaptative
        
        # Reshape pour GRU (ajouter dimension temporelle)
        y_compensated = y_compensated.unsqueeze(1)  # [batch_size, 1, n]
        
        gru_out, _ = self.gru(y_compensated)  # [batch_size, 1, hidden_dim]
        
        # Projection finale
        y = self.projection(gru_out.squeeze(1))  # [batch_size, m]
        
        return y