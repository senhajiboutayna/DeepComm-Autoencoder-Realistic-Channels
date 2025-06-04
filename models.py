import torch
import torch.nn as nn

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

class FeedbackCorrection2(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(FeedbackCorrection2, self).__init__()

        # Encoder profond avec skip connections
        self.encoder = nn.ModuleList()
        # Encoder adaptatif
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2)
        )
            
        # Bottleneck attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.Sigmoid()
        )
        
        # Decoder avec résidus
        self.decoder = nn.ModuleList()
        # Decoder avec skip connection
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # Gestion des dimensions
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Encoder
        x_enc = self.encoder(x)
        
        # Attention
        attn = self.attention(x_enc)
        x_attn = x_enc * attn
        
        # Decoder
        x_out = self.decoder(x_attn)
        
        # Restauration de la forme originale
        if len(original_shape) == 1:
            x_out = x_out.squeeze(0)
            
        return torch.tanh(x_out)