# 🧠 AI-Based Handwritten Digit Recognition System

An AI-powered handwritten digit recognition system built using **Convolutional Neural Networks (CNN)**, **TensorFlow/Keras**, **MNIST**, and **Streamlit**.

The system recognizes handwritten digits from **0 to 9** and provides prediction confidence, top predictions, model analytics, confusion matrix visualization, and error analysis through an interactive web interface.

---

## 🚀 Project Overview

Handwritten digit recognition is a fundamental computer vision and deep learning problem. This project develops a CNN-based artificial intelligence system capable of learning visual patterns from handwritten digits and classifying them into one of ten classes: **0–9**.

The model is trained and evaluated using the **MNIST handwritten digit dataset** and achieves a final test accuracy of:

> **99.46%**

The trained model is integrated into a Streamlit application that allows users to draw or upload handwritten digits and receive real-time predictions.

---

## ✨ Key Features

* 🔢 Recognition of handwritten digits from **0–9**
* 🧠 CNN-based deep learning model
* 📊 **99.46% test accuracy**
* ✍️ Draw digits directly using an interactive canvas
* 📤 Upload handwritten digit images
* 🎯 Prediction confidence score
* 🏆 Top-3 predicted digits
* 📈 Probability distribution visualization
* 🔍 Visualize processed input seen by the CNN
* 📊 Training and validation accuracy/loss curves
* 🧩 Confusion matrix
* 📋 Classification report
* ❌ Misclassification and error analysis
* 🌐 Interactive Streamlit web interface
* 💻 Deployable as a public web application

---

## 🛠️ Technologies Used

| Technology                | Purpose                 |
| ------------------------- | ----------------------- |
| Python                    | Programming language    |
| TensorFlow                | Deep learning framework |
| Keras                     | CNN model development   |
| NumPy                     | Numerical computation   |
| Pandas                    | Data processing         |
| OpenCV                    | Image preprocessing     |
| Pillow                    | Image handling          |
| Scikit-learn              | Model evaluation        |
| Matplotlib                | Visualization           |
| Streamlit                 | Web application         |
| Streamlit Drawable Canvas | Digit drawing interface |

---

## 📚 Dataset

The project uses the **MNIST handwritten digit dataset**.

### Dataset Information

* Training images: **60,000**
* Test images: **10,000**
* Image size: **28 × 28 pixels**
* Number of classes: **10**
* Classes: **0, 1, 2, 3, 4, 5, 6, 7, 8, 9**
* Original pixel range: **0–255**
* Normalized pixel range: **0–1**

The original training dataset was divided into:

* Training set: **54,000 images**
* Validation set: **6,000 images**
* Final test set: **10,000 images**

---

## 🔄 Data Preprocessing

The input images undergo preprocessing before being supplied to the CNN.

### Processing steps

1. Convert pixel values from `0–255` to `0–1`.
2. Convert images to floating-point representation.
3. Reshape images from:

```text
28 × 28
```

to:

```text
28 × 28 × 1
```

4. For real-world uploaded images:

   * Convert to grayscale
   * Normalize image intensity
   * Apply Gaussian blur
   * Perform thresholding
   * Remove small noise
   * Detect the foreground region
   * Crop the digit
   * Preserve aspect ratio
   * Resize the digit
   * Center it on a 28 × 28 canvas
   * Normalize before prediction

This preprocessing improves the model's ability to handle handwritten digits outside the original MNIST format.

---

## 🧠 CNN Architecture

The recognition model uses a convolutional neural network designed specifically for handwritten digit classification.

### Architecture

```text
Input
28 × 28 × 1

        ↓

Conv2D
32 Filters, 3 × 3
ReLU

        ↓

Batch Normalization

        ↓

Conv2D
32 Filters, 3 × 3
ReLU

        ↓

MaxPooling
2 × 2

        ↓

Dropout
0.25

        ↓

Conv2D
64 Filters, 3 × 3
ReLU

        ↓

Batch Normalization

        ↓

Conv2D
64 Filters, 3 × 3
ReLU

        ↓

MaxPooling
2 × 2

        ↓

Dropout
0.25

        ↓

Conv2D
128 Filters, 3 × 3
ReLU

        ↓

Batch Normalizati
```
