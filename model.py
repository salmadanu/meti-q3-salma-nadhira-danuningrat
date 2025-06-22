import torch
import torch.nn as nn

# Load latent dim if needed
def get_latent_dim(config_path="models/config.pth"):
    config = torch.load(config_path, map_location='cpu')
    return config.get('latent_dim', 20)

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + num_classes, 400)
        self.fc2 = nn.Linear(400, 28 * 28)

    def forward(self, z, y):
        z = torch.cat([z, y], dim=1)
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))