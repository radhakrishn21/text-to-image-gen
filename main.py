import torch
import matplotlib.pyplot as plt
from model import Generator
from text_utils import get_embedding

generator = Generator()

text = "a colorful object"
text_embedding = get_embedding(text).detach()

noise = torch.randn(1, 100)

generated_image = generator(text_embedding, noise).detach().numpy()

# Show image
plt.imshow(generated_image.reshape(28,28), cmap='gray')
plt.title(text)

# Save image
plt.savefig("output_image.png")

plt.show()