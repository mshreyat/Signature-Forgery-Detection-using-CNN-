# CNN-Based Offline Signature Forgery Detection

A deep learning-based system for detecting forged handwritten signatures using the **CEDAR Signature Dataset**. The project compares a custom Convolutional Neural Network (CNN) with a pretrained **MobileNetV2** transfer-learning model.

## 📌 Project Overview

Signature forgery detection is an important problem in document verification, banking, legal documentation, and authentication systems.

This project investigates whether deep learning can distinguish between:

- Genuine handwritten signatures
- Forged handwritten signatures

Two approaches are implemented and compared:

1. **Custom CNN** trained from scratch
2. **MobileNetV2** with a frozen ImageNet-pretrained base

The models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

---

## 🎯 Objectives

- Build an offline signature forgery detection system.
- Preprocess handwritten signature images using grayscale conversion and resizing.
- Train a CNN classifier from scratch.
- Apply transfer learning using MobileNetV2.
- Compare both approaches using standard classification metrics.
- Analyze the strengths and limitations of the models.

---

## 📂 Dataset

This project uses the **CEDAR Signature Dataset**.

The dataset contains:

- **1,320 genuine signatures**
- **1,320 forged signatures**
- **2,640 images in total**

The dataset is **not included in this repository**.

### Dataset Setup

Download the CEDAR dataset separately and place it in the project directory using the following structure:

```text
signature-forgery-detection/
│
├── miniproject_v2.ipynb
├── README.md
│
└── CEDAR/
    ├── full_org/
    └── full_forg/
