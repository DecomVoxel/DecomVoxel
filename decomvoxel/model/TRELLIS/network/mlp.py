import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=1):
        """
        Decoder-only MLP to output rotation angle.
        - input_dim: The size of the input latent vector.
        - hidden_dim: Number of neurons in hidden layers.
        - output_dim: The number of output parameters (1 for rotation angle).
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Latent code: Learnable input tensor
        self.latent = nn.Parameter(torch.randn(input_dim) * 0.1)

    def forward(self):
        x = self.latent  # Use the learnable latent vector as input
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))
        return output

if __name__ == '__main__':
    
    network = MLPDecoder()
    output = network()
    print(output)