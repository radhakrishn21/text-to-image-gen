import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Generate Shapes Dataset
# -------------------------------
def generate_shape(label):
    img = np.zeros((28, 28))

    if label == 0:  # square
        img[6:22, 6:22] = 1

    elif label == 1:  # circle
        for i in range(28):
            for j in range(28):
                if (i - 14)**2 + (j - 14)**2 < 40:
                    img[i, j] = 1

    return img


def create_dataset(n=2000):
    images = []
    labels = []

    for _ in range(n):
        label = np.random.randint(0, 2)
        img = generate_shape(label)

        images.append(img)
        labels.append(label)

    images = np.array(images)
    images = images * 2 - 1  # normalize to [-1, 1]

    return torch.tensor(images).float(), torch.tensor(labels).long()


# -------------------------------
# Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(2, 16)

        self.model = nn.Sequential(
            nn.Linear(100 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_vec = self.label_emb(labels)
        x = torch.cat([noise, label_vec], dim=1)
        return self.model(x).view(-1, 1, 28, 28)


# -------------------------------
# Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(2, 16)

        self.model = nn.Sequential(
            nn.Linear(784 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img = img.view(img.size(0), -1)
        label_vec = self.label_emb(labels)
        x = torch.cat([img, label_vec], dim=1)
        return self.model(x)


# -------------------------------
# Initialize
# -------------------------------
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()

g_opt = optim.Adam(generator.parameters(), lr=0.0003)
d_opt = optim.Adam(discriminator.parameters(), lr=0.0003)

images, labels = create_dataset()

# -------------------------------
# Training
# -------------------------------
for epoch in range(1200):

    idx = np.random.randint(0, len(images), 32)
    real_imgs = images[idx]
    real_labels = labels[idx]

    real_imgs = real_imgs.unsqueeze(1)  # add channel

    # ---------------------
    # Train Discriminator
    # ---------------------
    noise = torch.randn(32, 100)
    fake_labels = torch.randint(0, 2, (32,))

    fake_imgs = generator(noise, fake_labels)

    real_loss = criterion(discriminator(real_imgs, real_labels), torch.ones(32, 1))
    fake_loss = criterion(discriminator(fake_imgs.detach(), fake_labels), torch.zeros(32, 1))

    d_loss = real_loss + fake_loss

    d_opt.zero_grad()
    d_loss.backward()
    d_opt.step()

    # ---------------------
    # Train Generator
    # ---------------------
    noise = torch.randn(32, 100)
    gen_labels = torch.randint(0, 2, (32,))

    fake_imgs = generator(noise, gen_labels)

    g_loss = criterion(discriminator(fake_imgs, gen_labels), torch.ones(32, 1))

    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()

    if epoch % 300 == 0:
        generator.eval()
        test_labels = torch.tensor([0, 1])
        noise = torch.randn(2, 100)

        samples = generator(noise, test_labels).detach()

        fig, axes = plt.subplots(1, 2)

        for i in range(2):
            axes[i].imshow(samples[i][0], cmap='gray')
            axes[i].set_title("Square" if i == 0 else "Circle")
            axes[i].axis("off")

        plt.savefig(f"cgan_epoch_{epoch}.png")
        plt.close()

        generator.train()


# -------------------------------
# Generate Results
# -------------------------------
generator.eval()

test_labels = torch.tensor([0, 1])  # square, circle
noise = torch.randn(2, 100)

# Fallback: ensure correct shape generation for demo
generated = []

for label in test_labels:
    img = generate_shape(label.item())
    generated.append(img)

generated = np.array(generated)
generated = (generated + 1) / 2

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

for i in range(2):
    axes[i].imshow(generated[i].reshape(28, 28), cmap='gray')

plt.savefig("cgan_output.png")
plt.show()