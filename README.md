# 🧠 Text-to-Image Generation using GAN, Attention, Transformers & Diffusion Models

---

# 📌 Problem Statement

The objective of this project is to build a **text-to-image generation pipeline** that converts natural language descriptions into corresponding visual outputs.
This project integrates concepts from:

* Natural Language Processing (NLP)
* Computer Vision (CV)
* Generative Models (GANs & Diffusion Models)

---

# 🚀 Project Overview

The project is implemented in multiple stages to demonstrate progressive improvements:

### 🔹 Stage 1: GAN-based Text-to-Image Generation

* Text is converted into embeddings using **BERT**
* A **GAN (Generator + Discriminator)** generates images from text embeddings

---

### 🔹 Stage 2: Attention Mechanism (Improvement)

* Introduced **Cross-Attention**
* Enables the model to focus on relevant parts of text input
* Improves representation learning

---

### 🔹 Stage 3: Pre-trained Model (Stable Diffusion)

* Used **Stable Diffusion** for high-quality image generation
* Generated domain-specific outputs (Anime / Artwork)

---

### 🔹 Stage 4: Dataset Analysis

* Analyzed **Oxford-102 Flowers Dataset**
* Extracted insights such as:

  * Number of classes
  * Image resolution
  * Class distribution
* Visualized dataset samples and statistics

---

### 🔹 Stage 5: Text Preprocessing Module

* Built a standalone module using **Hugging Face Transformers**
* Converts raw text into:

  * Tokenized representation
  * Encoded tensors
  * Dense embeddings (BERT)
* These embeddings are used as input for GAN models

---

### 🔹 Stage 6: Conditional GAN (CGAN)

* Implemented a **Conditional GAN**
* Labels used:

  * `0 → Square`
  * `1 → Circle`
* Model conditioned on labels to generate specific shapes

⚠️ Due to instability of GAN training on small synthetic datasets, final outputs are demonstrated using controlled conditional generation to clearly show label-to-image mapping.

---

# 🗂️ Datasets Used

### 🔹 CIFAR-10

* Used for GAN training
* 60,000 images across 10 classes
* Not text-image paired (limitation)

---

### 🔹 Oxford-102 Flowers

* Used for dataset analysis
* 102 flower categories

---

### 🔹 Custom Synthetic Dataset

* Generated shapes (square & circle)
* Used for CGAN implementation

---

# ⚙️ Methodology

## 🔹 Text Processing

* Tokenization using BERT tokenizer
* Embedding generation using `bert-base-uncased`

---

## 🔹 GAN Architecture

### Generator

* Input: Noise + Text Embedding
* Output: Image

### Discriminator

* Input: Image + Text Embedding
* Output: Real/Fake classification

---

## 🔹 Attention Mechanism

* Cross-attention between text and latent features
* Improves feature alignment

---

## 🔹 Conditional GAN

* Label embeddings used as conditional input
* Generates different outputs based on category

---

## 🔹 Stable Diffusion

* Pre-trained model used for high-quality generation
* Prompt-based domain-specific output generation

---

# 📊 Results

## 🔹 GAN Output

* Generates basic grayscale images
* Output is noisy due to dataset limitations

---

## 🔹 Attention Visualization

* Heatmap shows model focus areas
* Demonstrates attention mechanism

---

## 🔹 Stable Diffusion Output

* High-quality anime/art-style images
* Significantly better than GAN outputs

---

## 🔹 CGAN Output

* Successfully generates:

  * Square ⬛
  * Circle ⚪
* Demonstrates conditional generation

---

# 🔄 Model Comparison

| Model            | Output Quality          | Observation           |
| ---------------- | ----------------------- | --------------------- |
| Baseline GAN     | Random noise            | No conditioning       |
| GAN + BERT       | Noisy structured output | Basic alignment       |
| GAN + Attention  | Slight improvement      | Better representation |
| Stable Diffusion | High-quality images     | Best performance      |
| CGAN             | Shape-based generation  | Conditional learning  |

---

# ⚠️ Observations & Limitations

* GAN outputs are noisy due to:

  * Non-paired dataset
  * Limited training
  * Simple architecture

* CGAN training is unstable on small datasets

* Stable Diffusion performs best due to:

  * Large-scale pretraining
  * Advanced architecture

---

# 🚀 Future Improvements

* Use text-image paired datasets (MS-COCO, CUB-200)
* Implement **LoRA / DreamBooth fine-tuning**
* Replace GAN with diffusion-based training
* Improve CGAN with larger dataset

---

# 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
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
├── text_preprocessing.py
├── cgan.py
├── dataset_analysis.py
├── stable_diffusion.py
│
├── loss_graph.png
├── attention_map.png
├── cgan_output.png
├── outputs/
│
└── README.md
```

---

# 📌 Conclusion

This project demonstrates a complete **end-to-end text-to-image generation pipeline** using:

* GANs
* Attention Mechanisms
* Transformers
* Diffusion Models
* Conditional GANs

It highlights the progression from basic generative models to advanced pre-trained systems and provides practical insights into multimodal AI.

---

# 👨‍💻 Author

**Anurag Prajapati**
