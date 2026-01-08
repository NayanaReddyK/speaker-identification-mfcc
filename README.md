# Speaker Identification using MFCCs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NayanaReddyK/speaker-identification-mfcc/blob/main/SIA_final_project.ipynb)

## Overview
This project implements an end-to-end **speaker identification system** using speech signal processing and machine learning techniques.  
The goal is to identify the speaker of a given audio sample based on acoustic characteristics extracted from speech.

---

## Dataset
- **AudioMNIST** dataset  
- Spoken digits (0–9) recorded by multiple speakers  
- Each speaker has multiple audio samples  
- Dataset link: https://github.com/soerenab/AudioMNIST  

---

## Approach

### Audio Preprocessing
- Resampling audio to 16 kHz
- Silence trimming
- Amplitude normalization

### Feature Extraction
- Mel-Frequency Cepstral Coefficients (MFCCs)
- 20 MFCC coefficients per frame
- Statistical aggregation for classical ML models
- Full MFCC time–frequency representations for deep learning models

---

## Models Implemented

### Classical Machine Learning
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM with RBF kernel)

### Deep Learning
- Baseline CNN on MFCC spectrograms
- Improved CNN with batch normalization and dropout
- CNN–LSTM hybrid architecture

---

## Evaluation
- Speaker-stratified train / validation / test split
- Metrics used:
  - Accuracy
  - Macro-averaged Precision
  - Recall
  - F1-score
- Confusion matrix analysis for class-wise performance

---

## Key Findings
- A tuned Support Vector Machine (RBF kernel) achieved the best overall performance (99.18% accuracy), outperforming both baseline and tuned deep learning models.
- Deep learning models benefited from architectural improvements (batch normalization, dropout, increased depth), but did not surpass classical ML on this dataset.
- The CNN–LSTM model underperformed due to the short duration and limited temporal variability of the speech samples, where explicit temporal modeling provided minimal benefit.
- Results highlight that increased model complexity does not necessarily improve performance when data is small, structured, and well-represented by engineered features.

---
## Key Takeaway
- For structured speech datasets with limited variability, strong feature engineering combined with classical machine learning can outperform more complex deep learning architectures.
---
## Tools & Libraries
- Python
- librosa
- NumPy, pandas
- scikit-learn
- TensorFlow / Keras
- Google Colab

---

## How to Run
1. Open the notebook using the **Open in Colab** badge above
2. Run the setup cells to install dependencies and download the dataset
3. Execute the notebook sequentially

---

## Author
Nayana Reddy
