# independent_study_homomorphic_encryption
Privacy-Preserving Chest X-Ray Classification using Homomorphic Encryption (Independent Study)

# Privacy-Preserving Chest X-Ray Classification using Homomorphic Encryption  
**Independent Study — University of Washington Tacoma (2025)**  
Author: **Aqueno Nirasmi Amalraj**

---

## Overview

This repository contains the code and artifacts for my independent study project on
**privacy-preserving neural network inference** using the CKKS **homomorphic encryption**
scheme.

The project focuses on a binary chest X-ray classification task (**NORMAL vs PNEUMONIA**):
a convolutional neural network (CNN) is trained in plaintext, and then the first fully
connected layer (FC1) is evaluated on **encrypted feature vectors** using the
[TenSEAL](https://github.com/OpenMined/TenSEAL) library.

The goal is to demonstrate that high-dimensional CNN features (length 50,176) can be
processed under homomorphic encryption while maintaining numerical fidelity, and to
quantify the runtime overhead compared to standard plaintext inference.

---

##  Project Structure

```text
independent_study_homomorphic_encryption/
│
├── Backup/                         # Older / experimental scripts (not used in final pipeline)
├── SC/                             # (Optional) Scratch or intermediate files
├── encrypted_eval_summary.csv      # Per-image metrics from multi-image encrypted evaluation
├── week2_cnn_train_improved.py     # Improved CNN training script (NORMAL vs PNEUMONIA)
├── evaluate_improved.py            # Plain test-set evaluation (accuracy, loss)
├── final_encryption.py             # Early single-ciphertext encryption experiments (prototype)
├── multi_cipher.py                 # Final multi-ciphertext encrypted FC1 inference (single image)
├── full_final.py                   # Encrypted FC1 evaluation on multiple images (3 NORMAL, 3 PNEUMONIA)
├── evaluation_encryption.py        # Script that runs multi-image encrypted evaluation and logs metrics
├── runtime_bar.py                  # Generates runtime comparison bar chart (Plain vs Encrypt vs Enc+Infer)
├── mse_per_image.py                # Generates MSE-per-image bar chart
├── corr_per_image.py               # Generates correlation-per-image bar chart
├── sample_images.py                # Script to generate NORMAL / PNEUMONIA sample figure
├── dataset_samples.png             # Sample NORMAL vs PNEUMONIA chest X-rays (used in report)
├── Baseline.png                    # Screenshot of plain CNN test accuracy and loss
├── SingleImage_multiCipher.png     # Console output for single-image multi-ciphertext experiment
├── Multi_image.png                 # Console output for 6-image encrypted evaluation
├── runtime_bar.png                 # Runtime comparison figure
├── mse_per_image.png               # MSE per image figure
├── corr_per_image.png              # Correlation per image figure
├── fc1_scatter.png                 # Scatter plot of plain vs encrypted FC1 activations
└── report/                         # (Optional) LaTeX source and PDF for the independent study report

Note: The actual chest X-ray dataset (e.g., data/chest_xray/) is not
included in this repository due to size and licensing constraints.

## Model and Dataset

Task: Binary classification – NORMAL vs PNEUMONIA

Input: Chest X-ray images resized to 224 × 224

Model: Custom CNN with:

3 convolutional blocks (Conv + BatchNorm + ReLU + MaxPool)

Flatten → FC1 (50,176 → 128) + ReLU + Dropout(0.5)

FC2 (128 → 2) for logits

Training:

Optimizer: Adam (lr = 8e-4, weight decay = 1e-5)

Weighted cross-entropy to handle class imbalance

15 epochs, batch size 16

Baseline performance on the held-out test set (624 images):

Test accuracy: 78.85%

Average test loss: 1.0767

(See Baseline.png and evaluate_improved.py.)

## Homomorphic Encryption Setup

Scheme: CKKS (approximate FHE)

Library: TenSEAL

Single-image multi-ciphertext experiment:

Feature vector length: 50,176

Chunk size: 8192 → 7 ciphertexts

Polynomial modulus degree: 16,384

Coefficient modulus bit sizes: [40, 21, 21, 40]

Global scale: 2²¹

The FC1 layer (Linear(50176 → 128)) is evaluated on encrypted features using a
multi-ciphertext dot product strategy.

## Key Experimental Results

For 6 test images (3 NORMAL, 3 PNEUMONIA), averaged metrics:

MSE between plain and encrypted FC1 outputs: ~456.99

Correlation between plain and encrypted FC1 outputs: 0.999824

Plain FC1 time: ~0.006 s per image

Encryption time: ~0.063 s per image

Encrypted FC1 inference time: ~29.38 s per image

Figures:

runtime_bar.png – Runtime comparison (log-scale)

mse_per_image.png – MSE per image

corr_per_image.png – Correlation per image

fc1_scatter.png – Scatter of plain vs encrypted FC1 activations (single image)

These results show that, although encrypted inference is much slower, the encrypted
outputs are numerically very close to the plaintext outputs (correlation > 0.9998).

## Environment and Installation

Recommended Python version: 3.10+

Install core dependencies:
pip install torch torchvision
pip install tenseal
pip install numpy matplotlib pillow

## How to Run
1. Train / load the CNN

If you already have a trained model file (week2_cnn_improved.pth), you can skip
re-training. Otherwise:
python week2_cnn_train_improved.py

2. Evaluate plain model on test set
python evaluate_improved.py

3. Single-image encrypted FC1 inference (multi-ciphertext)
python multi_cipher.py
This will:

Select a random PNEUMONIA test image

Extract the 50,176-dimensional feature vector

Encrypt it into 7 ciphertext chunks

Compute FC1 under encryption

Print MSE, correlation, and timings

Generate fc1_scatter.png (plain vs encrypted activations)

4. Multi-image encrypted evaluation
python full_final.py
This script:

Runs the encrypted FC1 layer on 6 test images (3 NORMAL, 3 PNEUMONIA)

Logs per-image metrics to the console

Saves a CSV summary (encrypted_eval_summary.csv)

5. Generate plots
python runtime_bar.py
python mse_per_image.py
python corr_per_image.py
python sample_images.py

Report

The accompanying LaTeX report, “Privacy-Preserving Chest X-Ray Classification Using Homomorphic Encryption”, summarizes:

CNN architecture and training

HE setup and CKKS parameter choices

Experimental design

Numerical fidelity results (MSE, correlation)

Runtime comparison and discussion

Limitations and future work

## Acknowledgments

This work was completed as an independent study under the supervision of
Prof. Dongfang Zhao, University of Washington Tacoma.
His guidance on homomorphic encryption, experimental design, and project scope is gratefully acknowledged.
