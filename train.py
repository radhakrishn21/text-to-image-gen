import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from text_utils import get_embedding

transform = transforms.Compose([
    transforms.Grayscale(),        # ✅ convert to 1 channel
    transforms.Resize((28, 28)),   # ✅ match model size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(root="./data", download=True, transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Loss function
criterion = nn.BCELoss()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

g_losses = []
d_losses = []

# Training loop
for epoch in range(10):
    for real_images, labels in dataloader:

        text = "a colorful object"
        text_embedding = get_embedding(text).detach()

        real_images = real_images.view(real_images.size(0), -1)
        noise = torch.randn(real_images.size(0), 100)

        # Train Discriminator
        fake_images = generator(text_embedding.repeat(real_images.size(0), 1), noise)

        real_label = torch.ones(real_images.size(0), 1)
        fake_label = torch.zeros(real_images.size(0), 1)

        d_loss_real = criterion(discriminator(real_images, text_embedding.repeat(real_images.size(0), 1)), real_label)
        d_loss_fake = criterion(discriminator(fake_images.detach(), text_embedding.repeat(real_images.size(0), 1)), fake_label)

        d_loss = d_loss_real + d_loss_fake

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        noise = torch.randn(real_images.size(0), 100)
        fake_images = generator(text_embedding.repeat(real_images.size(0), 1), noise)

        g_loss = criterion(discriminator(fake_images, text_embedding.repeat(real_images.size(0), 1)), real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())
    print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

import matplotlib.pyplot as plt

plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Graph")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()
    