# Stroke-Analysis
# SpikyUNet: Contrast CT Image Generation

This repository contains the **SpikyUNet deep learning model**, designed to generate **preprocessed contrast CT images** from **original NCCT images** and **preprocessed NCCT images**. The model is trained on the **publicly available UniToBrain dataset**, leveraging **hybrid loss (0.7 * MSE + 0.3 * SSIM)** and evaluates performance using **SSIM, PSNR, and MSE**.

---

## 📌 Features
- ✅ **DICOM Image Preprocessing** – Converts NCCT images to grayscale, applies CLAHE, and morphological operations  
- ✅ **Custom Hybrid Loss** – Combines MSE and SSIM for improved learning  
- ✅ **Metrics Tracking** – Computes **SSIM, PSNR, and MSE** during training, validation, and testing  
- ✅ **Dataset Splitting** – Ensures a balanced train/val/test split for effective learning  
- ✅ **Preprocessed and Original X Support** – Allows training on original NCCT or preprocessed NCCT images  

---

## 📌 Dataset: UniToBrain
In this work, we use the publicly available **UniToBrain dataset** 📄, which provides **NCCT images** and their corresponding **CBV perfusion maps**.  

🔗 **Dataset link:** [UniToBrain - IEEE Dataport](https://ieee-dataport.org/open-access/unitobrain)

### **📌 Dataset Statistics**
- **Total Samples:** **285**  
- **Filtered High-Quality Samples:** **258**  
- **Split Details:**
  - **Training:** **192 samples**
  - **Validation:** **49 samples**
  - **Testing:** **17 samples**
  
The dataset has been **carefully split** to ensure a **balanced approach** for **learning, hyperparameter tuning, and final evaluation**.

---

# 📌 Data Preparation

## 1️⃣ Dataset Format
The dataset consists of **NCCT images** as input (**X**) and **CBV perfusion maps** as output (**Y**).

- **X (Input):**
  - Can be **original NCCT images**
  - OR **preprocessed NCCT images** (CLAHE + Morphological operations)

- **Y (Output):**
  - Always **preprocessed contrast CT images (CBV maps)**

---

## 2️⃣ Preprocessing Options
SpikyUNet supports **both original and preprocessed NCCT images** as input.  
You can configure this setting in **`train.py`**:

```python
use_preprocessed_x = False  # Set True to use CLAHE + Morphology; False for original X
```

---

## 🚀 Training

Run the following command to **train the model**:  
The **X and Y splits for testing** are also saved while executing this code.

```bash
python train.py
```

---

## 🚀 Testing

Run the following command to **test the model**:  
- Calls the saved **test `.npy` files** while executing `train.py`.

```bash
python test.py
```

---

## 📂 Folder Structure
```
SpikyUNet/
│── preprocessing.py        # Preprocesses NCCT images
│── dataset.py              # Defines PyTorch dataset class
│── model.py                # SpikyUNet model architecture
│── loss.py                 # Custom hybrid loss function
│── train.py                # Training script (with validation)
│── test.py                 # Testing script (loads best model)
│── requirements.txt        # Required dependencies
│── README.md               # Setup and usage instructions
│── X_test.npy (Generated)  # Test dataset saved during training
│── Y_test.npy (Generated)  # Test dataset saved during training
│── best_model.pth (Saved)  # Best trained model checkpoint
```
