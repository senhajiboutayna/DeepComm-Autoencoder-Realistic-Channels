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
    def __init__(self, input_dim, hidden_dim=256, num_layers=3):
        super(FeedbackCorrection2, self).__init__()

        # Store original hidden_dim for decoder
        self.original_hidden_dim = hidden_dim

        # Encoder profond avec skip connections
        self.encoder = nn.ModuleList()
        current_dim = input_dim
        self.encoder_dims = []

        for i in range(num_layers):
            self.encoder.append(nn.Linear(current_dim, hidden_dim))
            self.encoder_dims.append(current_dim)
            self.encoder.append(nn.LayerNorm(hidden_dim))
            self.encoder.append(nn.LeakyReLU(0.2))
            current_dim = hidden_dim
            hidden_dim = max(hidden_dim//2, 32)  # Réduction progressive
            
        # Bottleneck attention
        self.attention = nn.Sequential(
            nn.Linear(current_dim, current_dim),
            nn.Sigmoid()
        )
        
        # Decoder avec résidus
        self.decoder = nn.ModuleList()
        self.decoder_dims = []

        # Reverse the hidden_dim progression for decoder
        hidden_dims = [current_dim]

        for i in range(num_layers-1):
            current_dim = min(current_dim*2, self.original_hidden_dim*2)
            hidden_dims.append(current_dim)
        hidden_dims = hidden_dims[::-1]  # Reverse for decoder
        
        for i, out_dim in enumerate(hidden_dims):
            in_dim = current_dim if i == 0 else hidden_dims[i-1]
            self.decoder.append(nn.Linear(in_dim, out_dim))
            self.decoder.append(nn.LayerNorm(out_dim))
            self.decoder.append(nn.LeakyReLU(0.2))
            self.decoder_dims.append(in_dim)  # Save input dimension
            current_dim = out_dim
            
        self.final = nn.Sequential(
            nn.Linear(current_dim, input_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Store residuals with their original dimensions
        residuals = []
        
        # Encoder pass
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                residuals.append(x)
            x = layer(x)
        
        # Attention bottleneck
        attn = self.attention(x)
        x = x * attn
        
        # Decoder pass with proper residual connections
        residual_ptr = len(residuals) - 1
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Linear) and residual_ptr >= 0:
                # Project residual if dimension doesn't match
                if residuals[residual_ptr].shape[-1] != x.shape[-1]:
                    residual = nn.Linear(residuals[residual_ptr].shape[-1], x.shape[-1]).to(x.device)(residuals[residual_ptr])
                else:
                    residual = residuals[residual_ptr]
                x = x + residual
                residual_ptr -= 1
            x = layer(x)
                
        return self.final(x)