import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from text_utils import get_embedding

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Data Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(root="./data", download=True, transform=transform)

# 👉 SPEED FIX: use small subset
dataset = torch.utils.data.Subset(dataset, range(2000))

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------------
# Models
# -------------------------------
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

g_losses = []
d_losses = []

# 👉 Store only ONE attention sample (FAST)
saved_attention = None

# -------------------------------
# Training Loop
# -------------------------------
for epoch in range(3):   # 👉 reduce epochs for speed
    for i, (real_images, _) in enumerate(dataloader):

        text = "a colorful object"
        text_embedding = get_embedding(text).detach()

        batch_size = real_images.size(0)

        real_images = real_images.view(batch_size, -1)
        text_embedding_batch = text_embedding.repeat(batch_size, 1)

        noise = torch.randn(batch_size, 100)

        # ---------------------
        # Capture Attention (ONLY ONCE)
        # ---------------------
        if saved_attention is None:
            attn_output = generator.attention(text_embedding_batch, text_embedding_batch)
            saved_attention = attn_output.detach().numpy()

        # ---------------------
        # Train Discriminator
        # ---------------------
        fake_images = generator(text_embedding_batch, noise)

        real_label = torch.ones(batch_size, 1)
        fake_label = torch.zeros(batch_size, 1)

        d_loss_real = criterion(discriminator(real_images, text_embedding_batch), real_label)
        d_loss_fake = criterion(discriminator(fake_images.detach(), text_embedding_batch), fake_label)

        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ---------------------
        # Train Generator
        # ---------------------
        noise = torch.randn(batch_size, 100)
        fake_images = generator(text_embedding_batch, noise)

        g_loss = criterion(discriminator(fake_images, text_embedding_batch), real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# -------------------------------
# Plot Loss Graph
# -------------------------------
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Graph")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()

# -------------------------------
# Attention Visualization (FAST)
# -------------------------------
avg_attention = saved_attention[0]  # take first sample

# Normalize
avg_attention = (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min())

# Reduce to 784
if avg_attention.shape[0] < 784:
    avg_attention = np.pad(avg_attention, (0, 784 - avg_attention.shape[0]))
else:
    avg_attention = avg_attention[:784]

plt.imshow(avg_attention.reshape(28, 28), cmap='hot')
plt.title("Attention Map Visualization")
plt.colorbar()

plt.savefig("attention_map.png")
plt.show()