import torch
from cleaner import  acceptable_phones

DATA_DIM = len(acceptable_phones)

class AE(torch.nn.Module):
    def __init__(self, HIDDEN_DIM):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(DATA_DIM, HIDDEN_DIM),
            torch.nn.Sigmoid()
        )
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_DIM, DATA_DIM),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
       return self.encoder(x)
    
    def decode(self,x):
        return self.decoder(x)