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
- Tuned **SVM with RBF kernel achieved the best performance**
- Deep learning models did not outperform classical ML due to the small and structured nature of the dataset
- Increasing model complexity does not necessarily improve performance when data is limited
- Model choice should align with dataset characteristics rather than architectural complexity

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
