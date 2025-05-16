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
        y = y / torch.norm(x, dim=1, keepdim=True) * np.sqrt(self.n)    # Power normalization
        """
        Currently, BatchNorm1d is used, but in a real case, the signal must respect a power constraint.
        Idea: Force an average power of 1 with explicit normalization
        This will ensure that the transmitter does not exceed the permitted power.
        """
        return y


class Decoder(nn.Module):

    def __init__(self, m, n, use_csi=False):

        super(Decoder, self).__init__()

        self.n = n
        self.use_csi = use_csi
        self.input_dim = n if not use_csi else n * 3  # +2 pour (Re(h), Im(h))

        self.linear_relu = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
        )
        
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=256, out_features=m),
            nn.LogSoftmax(dim=1),
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, y,  h=None):

        y = y.view(-1, self.n)  

        if self.use_csi and h is not None:
            # Assurez-vous que h est 2D : (batch_size, n)
            if h.dim() == 1:
                h = h.unsqueeze(0)  # (n,) -> (1, n)
            elif h.dim() == 0:
                raise ValueError("h should be at least 1D tensor.")

            if h.shape[0] == 1 and y.shape[0] > 1:
                h = h.expand(y.shape[0], -1)
                
            # Normalize CSI properly
            h_norm = h / (torch.norm(h, p=2, dim=1, keepdim=True) + 1e-6)
            
            # Concatenate features
            y = torch.cat([y, h, h_norm], dim=1)


        # Decoding phase
        y = self.linear_relu(y)
        y = self.linear_out(y)
        
        return y
    

class FeedbackCorrection(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(FeedbackCorrection, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, input_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Add temporal dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        lstm_out = lstm_out[:, -1, :]
        y = self.fc(lstm_out)

        return y