# Text-to-Image Generation using GAN and BERT

# Problem Statement

The goal of this project is to build a **text-to-image generation pipeline** that converts natural language descriptions into corresponding images.
This simulates a real-world AI application combining **Natural Language Processing (NLP)** and **Computer Vision (CV)**.

---

# Project Overview

This project implements a pipeline where:

1. Text input is processed and converted into embeddings using **BERT**
2. A **Generative Adversarial Network (GAN)** generates images based on text embeddings
3. The generated output is visualized and evaluated

---

# Dataset

* Dataset used: **CIFAR-10**
* Contains 60,000 images across 10 classes
* Note: This dataset is **not text-image paired**, so text conditioning is simulated

---

# Methodology

### 🔹 1. Text Preprocessing

* Input text is tokenized using **BERT Tokenizer**
* Converted into dense vector embeddings using **BERT model**

---

### 🔹 2. Text Embedding

* Used: `bert-base-uncased`
* Output: Fixed-size vector representation of input text
* Purpose: Capture semantic meaning of text

---

### 🔹 3. GAN Architecture

#### Generator

* Inputs: Noise vector + Text embedding
* Output: Generated image
* Goal: Produce realistic images from text

#### Discriminator

* Inputs: Image + Text embedding
* Output: Real/Fake classification
* Goal: Distinguish real vs generated images

---

### 🔹 4. Training Process

* Loss Function: Binary Cross Entropy (BCE Loss)
* Optimizer: Adam
* Training involves:

  * Updating discriminator with real and fake images
  * Updating generator to fool discriminator

---

# Results

### 🔹 Loss Graph

* Generator and discriminator losses were plotted over epochs
* Shows adversarial training behavior

### 🔹 Generated Output

* The model generates grayscale images based on input text

---

# Observations & Insights

* Generated images appear **noisy**
* Reason:

  * CIFAR-10 is not text-image paired dataset
  * Limited training epochs
  * Basic GAN architecture

---

# Model Comparison

| Model     | Description                                     |
| --------- | ----------------------------------------------- |
| Baseline  | Random noise (no conditioning)                  |
| GAN Model | Generates images conditioned on text embeddings |

👉 The GAN model performs better than baseline by incorporating semantic information.

---

# Future Improvements

* Use text-image paired datasets (e.g., **MS-COCO, CUB-200**)
* Replace GAN with **Diffusion Models**
* Use **CLIP embeddings** for better alignment
* Train for more epochs with better architecture

---

# Tech Stack

* Python
* PyTorch
* Transformers (HuggingFace)
* Matplotlib

---

## 📁 Project Structure

```
text_to_image/
│
├── train.py           # Training loop
├── model.py           # GAN architecture
├── text_utils.py      # Text processing & embeddings
├── main.py            # Image generation script
├── loss_graph.png     # Training visualization
├── output_image.png   # Generated output
└── README.md
```

---

# Conclusion

This project demonstrates an end-to-end pipeline integrating NLP and GANs for text-to-image generation.
Although results are basic, the system successfully shows how multimodal AI models can be built.


# Author

Anurag Prajapati
