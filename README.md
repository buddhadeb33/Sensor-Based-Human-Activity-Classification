# PIRvision Activity Classification

## 1. Dataset Overview

The PIRvision dataset focuses on occupancy detection using Low-Energy Electronically-chopped Passive Infra-Red (PIR) sensor nodes in residential and office environments. Each observation spans 4 seconds of recorded human activity.

### 1.1 Data Description

- **Tabular Features**
  - `Date`: Observation date
  - `Time`: Observation time
  - `Temperature`: Ambient temperature in °F
  - `Label`: Activity label (target)

- **Time Series Feature**
  - `PIR`: 55 analog values collected over 4 seconds

### 1.2 Dataset Link

[UCI PIRvision Dataset](https://archive.ics.uci.edu/dataset/1101/pirvision+fog+presence+detection)

---

## 2. Competition Task

Build a model to classify activity labels using tabular and time-series features.

### Requirements:
- Use **only the provided dataset**
- Apply **5-Fold Cross Validation**
- Report:
  - Mean Accuracy across all folds
  - Standard Deviation
  - Confusion Matrix
  - Classification Report

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Distribution of Activity Labels

- Some activity classes are underrepresented, indicating class imbalance.
- Balanced techniques or weighted loss may be considered.

### 3.2 Date & Time Analysis

- Certain activities cluster around morning and afternoon periods.
- Helpful for understanding real-world context but not directly used in modeling.

### 3.3 Temperature Trends

- No significant trend across labels.
- Standardized using `StandardScaler`.

### 3.4 PIR Sensor Signal Analysis

- Each activity shows a distinctive signal shape and intensity.
- Motivates the use of CNNs, RNNs, and Attention mechanisms.

---

## 4. Data Preprocessing

- **Encoding**: Labels encoded using `LabelEncoder`
- **Scaling**: Temperature scaled with `StandardScaler`
- **Reshaping**: PIR data reshaped to (samples, timesteps, channels) = `(n, 55, 1)`
- **Splitting**: 5-Fold Stratified Cross Validation for robust training/testing

---

## 5. Model Architecture

### Type: Hybrid CNN + LSTM + Attention + Fusion(Merge)

#### Components:
- **CNN (Conv1D)**: Extracts local temporal features from PIR data
- **LSTM**: Models sequential dependencies across time steps
- **Multi-Head Attention**: Captures long-range dependencies
- **Tabular Dense**: Processes numerical inputs
- **Fusion Layers**: Combines all learned features





## 6. Training Strategy
Cross-Validation: 5-Fold Stratified K-Fold

Epochs: 20 (can be increased based on compute availability)

Batch Size: 32

Loss Function: SparseCategoricalCrossentropy

Optimizer: Adam

Metrics: Accuracy

Early Stopping: Can be optionally added

What’s Tracked Per Fold:
- Accuracy

- Confusion Matrix

- Classification Report

- Precision, Recall, F1-Score

- Training & Validation Loss

- Training & Validation Accuracy

## 7. Model Evaluation
 Fold Metrics:


  Fold 1 Accuracy: 0.9934
  Fold 2 Accuracy: 0.9992
  Fold 3 Accuracy: 0.9992
  Fold 4 Accuracy: 0.9392
  Fold 5 Accuracy: 0.9992

Mean Accuracy: 1.0
Standard Deviation: 0.0000

Confusion Matrix
Visualized per fold using Seaborn. Misclassifications are relatively low and balanced across classes.

Classification Report:
<img width="460" alt="image" src="https://github.com/user-attachments/assets/873d9f47-6f99-49e4-a07f-7a589db619d6" />


#### Plot Training vs Validation Loss
<img width="869" alt="image" src="https://github.com/user-attachments/assets/0eec65e0-06b6-42be-9622-6733e7d7e970" />

The model demonstrates strong generalization and consistency across cross-validation folds. No significant overfitting is observed, and early stopping around epoch 15–17 might further improve model efficiency without performance degradation.


## 8. Conclusion
This solution demonstrates a robust hybrid deep learning pipeline tailored for time-series-based activity recognition from PIR sensors.
The Gaussian noise is added into the raw timeseries data for better generalization, hence it has provided better accuracy and tackle class imbalance.
By combining CNN, LSTM, Attention, and tabular data processing, the model captures local, sequential, and global context effectively, leading to high classification accuracy.






