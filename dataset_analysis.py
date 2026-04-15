import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# -------------------------------
# Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------------------------------
# Load Dataset
# -------------------------------
dataset = datasets.Flowers102(root="./data", download=True, transform=transform)

print("Total images:", len(dataset))
print("Number of classes:", len(dataset.classes))

# -------------------------------
# SHOW SAMPLE IMAGES
# -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

for i, ax in enumerate(axes.flatten()):
    image, label = dataset[i]
    image = image.permute(1, 2, 0)
    
    ax.imshow(image)
    ax.set_title(f"Flower: {dataset.classes[label]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("dataset_samples.png")
plt.show()

# -------------------------------
# IMAGE RESOLUTION ANALYSIS
# -------------------------------
heights = []
widths = []

for i in range(50):  # sample first 50 images
    img, _ = dataset[i]
    heights.append(img.shape[1])
    widths.append(img.shape[2])

print("Average Height:", np.mean(heights))
print("Average Width:", np.mean(widths))

# -------------------------------
# CLASS DISTRIBUTION
# -------------------------------
labels = [dataset[i][1] for i in range(len(dataset))]
count = Counter(labels)

# Plot distribution
plt.figure()
plt.bar(count.keys(), count.values())
plt.xlabel("Class Index")
plt.ylabel("Number of Images")
plt.title("Class Distribution")

plt.savefig("class_distribution.png")
plt.show()