import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import math

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

    def forward(self, y, chann_type="AWGN"):

        y = y.view(-1, self.n)  

        # Decoding phase
        y = self.linear_relu(y)
        y = self.linear_out(y)
        
        return y
