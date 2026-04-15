import torch
import torch.nn as nn

# -------------------------------
# Cross Attention Module
# -------------------------------
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_embed, noise_embed):
        # Project
        Q = self.query(noise_embed)
        K = self.key(text_embed)
        V = self.value(text_embed)

        # Compute attention (element-wise for simplicity)
        attention_scores = torch.softmax(Q * K / (self.embed_dim ** 0.5), dim=1)

        # Apply attention
        out = attention_scores * V
        return out


# -------------------------------
# Generator with Attention
# -------------------------------
class Generator(nn.Module):
    def __init__(self, text_dim=768, noise_dim=100):
        super().__init__()

        self.attention = CrossAttention(text_dim)

        self.model = nn.Sequential(
            nn.Linear(text_dim + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, text_embedding, noise):
        # Apply attention
        attn_output = self.attention(text_embedding, text_embedding)

        # Combine with noise
        x = torch.cat((attn_output, noise), dim=1)

        return self.model(x)


# -------------------------------
# Discriminator
# -------------------------------
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