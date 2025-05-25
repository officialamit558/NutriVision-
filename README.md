# 🥗 NutriVision: Food Classification & Nutrition Estimation using Vision Transformer (ViT)

NutriVision is a deep learning project that uses Vision Transformers (ViT) to classify food images and predict their nutritional content. By leveraging the power of Transformer-based architectures, the model achieves high accuracy on food image datasets and helps estimate nutritional values from images.
### Deployed App :- https://nutrivision558.streamlit.app/
---
### Web Interface
![Input Image](https://github.com/officialamit558/NutriVision/blob/main/Screenshot%202025-05-12%20103357.png)
![Analysis Results](https://github.com/officialamit558/NutriVision/blob/main/Screenshot%202025-05-12%20104852.png)
![Not Food Item Detected](https://github.com/officialamit558/NutriVision/blob/main/Screenshot%202025-05-12%20103523.png)

## 📁 Dataset

- **Dataset Used**: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)  
- **Description**:  
  - Contains **101 food categories** and **101,000 images**.
  - Images are split into **75,750 for training** and **25,250 for testing**.
  - Dataset size: ~5 GB.
- **Preprocessing**:
  - Created `train_dataloader` and `test_dataloader`.
  - Used **batch size = 32**, **epochs = 10**.
  - Applied standard image augmentation techniques (resize, normalize, etc.).

---
## 🔧 Hardware & Training Setup

To train the Vision Transformer model efficiently, the following hardware and software setup was used:

### 🖥️ GPU Specifications

| Parameter         | Value                            |
|------------------|----------------------------------|
| GPU Model         | NVIDIA GeForce RTX 4060          |
| CUDA Version      | 12.6                             |
| Driver Version    | 561.09                           |
| VRAM Usage        | 7244 MiB / 8188 MiB              |
| Power Draw        | 77W / 80W                        |
| GPU Utilization   | 100%                             |
| Operating System  | Windows with WDDM Driver Model   |

![GPU Usage](https://github.com/officialamit558/NutriVision/blob/main/GPU_Used.png)

*Note: The model was trained using a single RTX 4060 GPU. The high utilization (100%) indicates efficient usage of hardware resources during training.*

---

### ⏱️ Training Time & Efficiency

| Parameter             | Value            |
|----------------------|------------------|
| Total Training Epochs | 10               |
| Batch Size            | 32               |
| Training Time (approx)| ~6 Hours      |
| Framework Used        | PyTorch          |

The training process was optimized using mixed precision training and efficient data loading with PyTorch `DataLoader`.

---


## 🧠 Model Architecture

This project implements the **Vision Transformer (ViT)** architecture from scratch using PyTorch.

### 🔍 Paper Reference

> **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**  
> [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)

---

### ⚙️ Key Components of ViT

#### 1. **Image Patch Embedding**
- Each input image is divided into **fixed-size patches** (e.g., 16x16).
- Patches are **flattened** and passed through a **linear projection layer**.
- A learnable **[CLS] token** is prepended to the patch sequence.

#### 2. **Position Embeddings**
- Positional encodings are added to patch embeddings to retain spatial information.

#### 3. **Transformer Encoder Blocks**
Each block consists of:
- **LayerNorm**
- **Multi-Head Self Attention**
- **Skip/Residual Connections**
- **Feedforward MLP layers**
- **Repeated L times** (e.g., L = 12)

#### 4. **Classification Head**
- The final embedding of the `[CLS]` token is sent through an **MLP head** to predict food class.

---

## 🖼️ ViT Architecture Diagram

![ViT Architecture](https://github.com/officialamit558/NutriVision/blob/main/ViT.png)
*Source: ViT Paper (Dosovitskiy et al., 2020)*

---

## 🧪 Performance & Results

| Metric           | Value        |
|------------------|--------------|
| Accuracy         | ~51 % |
| Loss             | Cross-Entropy |
| Optimizer        | Adam |
| Learning Rate    | 1e-3 |

---

## ✅ Features

- 📦 End-to-end ViT implementation in PyTorch  
- 🍱 Food classification into 101 categories  
- 🔢 Nutrient prediction (optional extension)  
- 🧪 Custom training & evaluation loops  

---

## 📌 Future Work

- Extend the model to estimate **macronutrients** (carbs, fats, proteins) using metadata or additional models.
- Experiment with **larger ViT variants** (ViT-L, ViT-H) and pretrained weights.
- Deploy the model using **Streamlit or FastAPI** for real-time inference.

---

## 📜 Citation

If you use this work or are inspired by it, please cite the original ViT paper:

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk et al.},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}

