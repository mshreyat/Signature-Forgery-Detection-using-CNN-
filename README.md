# Signature Forgery Detection using CNN

## Overview

This project uses a Convolutional Neural Network (CNN) to classify handwritten signature images as **genuine** or **forged**. It uses the CEDAR Signature Dataset and demonstrates image preprocessing, model training, evaluation, and individual signature prediction.

## Problem Statement

Handwritten signatures are commonly used for identity verification. Manual verification can be time-consuming and subjective. This project explores how deep learning can assist offline signature verification by learning visual patterns from signature images.

## Objectives

- Classify signature images into genuine and forged categories.
- Preprocess images for CNN input.
- Train and evaluate a CNN model.
- Measure performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Predict the class of an individual signature image.

## Dataset

**CEDAR Signature Dataset**

The project uses genuine and skilled-forgery signature images. The dataset folders used in this implementation are:

```text
CEDAR/
├── full_org/   # Genuine signatures
└── full_forg/  # Forged signatures
```

The standard dataset is commonly described as containing 1,320 genuine and 1,320 forged images. Verify the counts in your downloaded copy before reporting them. The dataset is not included in this repository; obtain it from an authorized source and follow its terms of use.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- scikit-learn
- Seaborn

## Methodology

1. Load genuine and forged signature images.
2. Convert images to grayscale.
3. Resize images to 128 × 64 pixels.
4. Normalize pixel values to the range 0–1.
5. Assign labels: `0 = Genuine`, `1 = Forged`.
6. Split the dataset into training and testing sets using an 80:20 split.
7. Reshape images to include the grayscale channel.
8. Build and train a CNN.
9. Evaluate the model on the test set.
10. Generate a confusion matrix and classification report.
11. Test an individual signature image.

## CNN Architecture

The model contains:

- Convolutional layers with ReLU activation
- Max-pooling layers
- A flattening layer
- A dense layer with ReLU activation
- Dropout for regularization
- A sigmoid output layer for binary classification

The model is compiled using the Adam optimizer and binary cross-entropy loss.

## Results

Add the actual results from your notebook after training. Do not use example values.

| Metric | Result |
|---|---|
| Test accuracy | YOUR_TEST_ACCURACY |
| Genuine precision | YOUR_GENUINE_PRECISION |
| Genuine recall | YOUR_GENUINE_RECALL |
| Forged precision | YOUR_FORGED_PRECISION |
| Forged recall | YOUR_FORGED_RECALL |
| Forged F1-score | YOUR_FORGED_F1_SCORE |

Include your confusion matrix and training/validation accuracy and loss plots in this section if available.

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

### 2. Install dependencies

```bash
pip install tensorflow numpy pillow matplotlib scikit-learn seaborn
```

### 3. Download the dataset

Download the CEDAR Signature Dataset separately and arrange it as shown in the Dataset section. Update the dataset paths in the notebook if your folder structure differs.

### 4. Run the notebook

Open the project notebook in Jupyter Notebook or VS Code and run the cells in order, from preprocessing through evaluation and prediction.

## Limitations

- The model's performance depends on dataset quality, preprocessing, and the train/test split.
- A random image-level split may place signatures from the same writer in both training and testing sets. This can overestimate performance on signatures from previously unseen writers.
- A CNN prediction is not proof that a signature is authentic. This project is educational and should not be used as the sole basis for financial, legal, or identity-verification decisions.

## Future Improvements

- Use writer-independent train/test evaluation.
- Apply data augmentation carefully.
- Experiment with Siamese networks for writer-specific signature verification.
- Build a web interface for uploading and checking signature images.
- Evaluate on additional signature datasets.

## Conclusion

This project demonstrates how a CNN can be applied to offline handwritten signature classification. The workflow covers image preprocessing, supervised learning, and performance evaluation. The results should be interpreted in light of the dataset and evaluation method, and further testing is required before real-world use.

## Author

YOUR_NAME

## Disclaimer

This project is developed for educational purposes only. It is not a certified signature-authentication system.
