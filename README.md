# 🧠 Text-to-Image Generation using GAN, BERT, Attention & Stable Diffusion

---

# 📌 Problem Statement

The objective of this project is to build a **text-to-image generation system** that converts natural language descriptions into corresponding images.

This project simulates a real-world AI pipeline by integrating:

* **Natural Language Processing (NLP)**
* **Computer Vision (CV)**
* **Generative Models (GAN & Diffusion Models)**

---

# 🚀 Project Overview

This project is divided into four stages:

### 🔹 Stage 1: GAN-based Text-to-Image

* Text is converted into embeddings using **BERT**
* A **Generative Adversarial Network (GAN)** generates images

### 🔹 Stage 2: Attention-based Improvement

* Introduced **Cross-Attention mechanism**
* Helps model focus on relevant features from text

### 🔹 Stage 3: Pre-trained Model (Stable Diffusion)

* Used **Stable Diffusion v1.5**
* Generated domain-specific images (Anime/Artwork)

### 🔹 Stage 4: Dataset Analysis & Exploration

* Explored **Oxford-102 Flowers Dataset**
* Analyzed dataset structure, classes, and image properties
* Visualized images with labels and distribution

---

# 🗂️ Datasets Used

### 🔹 CIFAR-10 (Primary Dataset)

* 60,000 images across 10 classes
* Used for GAN training
* ⚠️ Not text-image paired → limits output quality

### 🔹 Custom Anime Dataset

* Small dataset for domain-specific generation
* Used for Stable Diffusion prompt conditioning

### 🔹 Oxford-102 Flowers Dataset

* Used for dataset exploration task
* Contains multiple flower categories with labeled images

---

# ⚙️ Methodology

## 🔹 1. Text Preprocessing

* Tokenization using **BERT Tokenizer**
* Conversion into embeddings using **BERT model**

---

## 🔹 2. Text Embedding

* Model: `bert-base-uncased`
* Generates semantic vector representations of text

---

## 🔹 3. GAN Architecture

### Generator

* Input: Noise + Text embedding
* Output: Generated image

### Discriminator

* Input: Image + Text embedding
* Output: Real/Fake classification

---

## 🔹 4. Attention Mechanism

* Implemented **Cross-Attention**
* Improves interaction between text and image features
* Enables better feature alignment

---

## 🔹 5. Stable Diffusion (Pre-trained Model)

* Used pre-trained **Stable Diffusion v1.5**
* Generated high-quality anime-style images
* Demonstrated domain-specific control using prompts

---

## 🔹 6. Dataset Analysis

* Loaded Oxford-102 dataset
* Analyzed:

  * Number of classes
  * Image resolution
  * Class distribution
* Visualized:

  * Sample images with labels
  * Distribution graph

---

# 📊 Results

## 🔹 GAN Training

* Generated grayscale images
* Loss graph shows adversarial training behavior

## 🔹 Attention Visualization

* Heatmap displays model focus areas
* Confirms attention mechanism is working

## 🔹 Stable Diffusion Outputs

* High-quality anime images generated
* Significant improvement over GAN outputs

## 🔹 Dataset Analysis

* Sample images visualized with labels
* Class distribution plotted
* Image resolution analyzed

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
  * Basic architecture

* Attention improves internal representation but has limited visual impact

* Stable Diffusion performs best due to:

  * Large-scale pre-training
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
├── dataset_analysis.py
├── loss_graph.png
├── output_image.png
├── attention_map.png
├── dataset_samples.png
├── class_distribution.png
├── outputs/
└── README.md
```

---

# 📌 Conclusion

This project demonstrates an end-to-end **multimodal AI pipeline** integrating:

* GAN-based image generation
* Attention mechanisms
* Pre-trained diffusion models
* Dataset exploration techniques

While GAN outputs are limited, the addition of attention and Stable Diffusion highlights the transition from basic to advanced generative AI systems.

---

# 👨‍💻 Author

**Anurag Prajapati**
