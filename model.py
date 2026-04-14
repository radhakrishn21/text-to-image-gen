import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, text_dim=768, noise_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(text_dim + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, text_embedding, noise):
        x = torch.cat((text_embedding, noise), dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, text_dim=768):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784 + text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, image, text_embedding):
        x = torch.cat((image, text_embedding), dim=1)
        return self.model(x)