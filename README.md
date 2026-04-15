# 🧠 Text-to-Image Generation using GAN, BERT, Attention & Stable Diffusion

---

# 📌 Problem Statement

The goal of this project is to build a **text-to-image generation pipeline** that converts natural language descriptions into corresponding images.
This project simulates a real-world AI system by combining:

* **Natural Language Processing (NLP)**
* **Computer Vision (CV)**
* **Generative Models (GAN & Diffusion Models)**

---

# 🚀 Project Overview

This project is implemented in three stages:

### 🔹 Stage 1: GAN-based Text-to-Image

* Text is converted into embeddings using **BERT**
* A **GAN (Generator + Discriminator)** generates images

### 🔹 Stage 2: Attention-based Improvement

* Introduced **Cross-Attention mechanism**
* Helps model focus on important features from text

### 🔹 Stage 3: Pre-trained Model (Stable Diffusion)

* Used **Stable Diffusion** for high-quality image generation
* Generated domain-specific outputs (Anime/Artwork)

---

# 🗂️ Dataset

### 🔹 CIFAR-10 (Primary Dataset)

* 60,000 images across 10 classes
* Used for GAN training
* ⚠️ Not text-image paired → limits output quality

### 🔹 Custom Dataset (Anime Images)

* Small dataset created for domain-specific generation
* Used to simulate fine-tuning of pre-trained model

---

# ⚙️ Methodology

## 🔹 1. Text Preprocessing

* Tokenization using **BERT Tokenizer**
* Conversion into embeddings using **BERT model**

---

## 🔹 2. Text Embedding

* Model: `bert-base-uncased`
* Converts text into meaningful vector representation

---

## 🔹 3. GAN Architecture

### Generator

* Input: Noise + Text embedding
* Output: Generated image

### Discriminator

* Input: Image + Text embedding
* Output: Real/Fake classification

---

## 🔹 4. Attention Mechanism (Improvement)

* Implemented **Cross-Attention**
* Enhances interaction between text and image generation
* Helps model focus on important features

---

## 🔹 5. Stable Diffusion (Pre-trained Model)

* Used pre-trained **Stable Diffusion v1.5**
* Generated anime-style images using prompts
* Demonstrates domain-specific image generation

---

# 📊 Results

## 🔹 Loss Graph

* Shows training behavior of Generator vs Discriminator

## 🔹 Generated GAN Output

* Produces grayscale images (noisy due to dataset limitation)

## 🔹 Attention Visualization

* Heatmap showing model focus regions
* Demonstrates attention mechanism working

## 🔹 Stable Diffusion Outputs

* High-quality anime images generated
* Shows improvement over GAN outputs

---

# 🔄 Model Comparison

| Model            | Output Quality          | Observation              |
| ---------------- | ----------------------- | ------------------------ |
| Baseline GAN     | Random noise            | No text alignment        |
| GAN + BERT       | Noisy structured output | Basic conditioning       |
| GAN + Attention  | Slightly improved focus | Better feature alignment |
| Stable Diffusion | High-quality images     | Best performance         |

---

# ⚠️ Observations & Insights

* GAN outputs are noisy due to:

  * Non-paired dataset (CIFAR-10)
  * Limited training
  * Simple architecture

* Attention improves internal representation but not visual quality significantly

* Stable Diffusion produces superior results due to:

  * Pre-training on large datasets
  * Advanced architecture

---

# 🚀 Pre-trained Model Refinement (Stable Diffusion)

A pre-trained Stable Diffusion model was used to generate anime-style images. A custom dataset of anime images was prepared to guide domain-specific generation.

Due to limited computational resources, full fine-tuning was approximated using **prompt engineering and controlled generation techniques**. The model successfully produced domain-specific outputs such as anime characters, cyberpunk scenes, and fantasy artwork.

---

# 🛠️ Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* Diffusers (Stable Diffusion)
* Matplotlib

---

# 📁 Project Structure

```
text_to_image/
│
├── train.py
├── model.py
├── text_utils.py
├── main.py
├── stable_diffusion.py
├── loss_graph.png
├── output_image.png
├── attention_map.png
├── outputs/                # Stable diffusion outputs
└── README.md
```

---

# 📌 Conclusion

This project demonstrates a complete pipeline for text-to-image generation using:

* GAN
* Attention Mechanisms
* Pre-trained Diffusion Models

While GAN-based outputs are limited, the integration of attention and Stable Diffusion showcases the evolution from basic to advanced generative AI systems.

---

# 👨‍💻 Author

**Anurag Prajapati**
