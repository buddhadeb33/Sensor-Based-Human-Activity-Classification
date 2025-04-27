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


---

# 📜 **Model Architecture Overview**

This model is a **multi-input deep learning model** that combines:

- **CNN** for local feature extraction from PIR sensor time-series.
- **LSTM + Transformer Attention** for capturing temporal dependencies in the PIR sequence.
- **Dense (MLP)** for handling auxiliary **tabular features** (like room temperature, humidity, etc.).
- **Merged Fusion** of all streams to perform final classification.

The purpose is to detect the presence or absence of a person (or other fine-grained activities) from **PIR + tabular features**.

---

# 🏛 **Detailed Component Breakdown**

## 1. **Inputs**
- **PIR Input:** `(timesteps, 1)` shaped sequence (e.g., 50 timestamps with 1 PIR sensor reading each).
- **Tabular Input:** `(3,)` shaped vector (e.g., 3 features like temperature, humidity, or other environmental factors).

---

## 2. **PIR Processing (Time Series Pathway)**

### ➡️ CNN Branch
- **Conv1D Layer (64 filters, kernel size=3):** Captures **local patterns** (small changes over adjacent PIR values).
- **MaxPooling1D:** Downsamples the time dimension to reduce computation and learn more abstract features.
- **Conv1D Layer (128 filters, kernel size=3):** Extracts **higher-level patterns** from already pooled features.
- **MaxPooling1D:** Further reduces dimensionality.
- **Flatten:** Prepares CNN output for merging later.

> This CNN branch **extracts spatial features** from the PIR sequence.

---

### ➡️ LSTM + Transformer Branch
- **First LSTM (64 units, `return_sequences=True`):** Captures temporal dependencies across PIR readings.
- **Second LSTM (32 units):** Summarizes sequence into a **compact temporal encoding**.
- **Multi-Head Attention (2 heads):** Provides **attention over time** — learns where in the sequence important changes occur.
- **Add & LayerNormalization:** Residual connection to stabilize training.
- **Flatten:** Prepares attention output for fusion.

> This LSTM + Transformer branch **captures sequence dynamics + focuses on important timestamps**.

---

## 3. **Tabular Data Processing**
- **Dense(16 units, ReLU activation):** Small neural net to learn embeddings from auxiliary tabular features.

> This branch **processes static features** that complement the dynamic PIR sensor readings.

---

## 4. **Merging Streams**
- **Concatenate:** CNN features + LSTM output + Attention output + Tabular features.
- **Dense(128 units + Dropout 0.3):** Fully connected layer to blend features together.
- **Dense(64 units + Dropout 0.3):** Further compression and regularization.
- **Final Dense layer:** Softmax layer for multi-class classification (`len(np.unique(y))` classes, like No Presence, Light Presence, Heavy Movement).

---

# ⚙️ **Training Details**
- **Loss Function:** `sparse_categorical_crossentropy` (labels are integer encoded, not one-hot).
- **Optimizer:** Adam.
- **Metrics:** Accuracy.

---

# 🧠 **Why This Architecture is Smart for FoG Presence Detection?**
- **CNN** quickly detects sharp changes in PIR activity (someone moving across the field).
- **LSTM** captures the sequential nature of presence (e.g., someone slowly moving or standing still).
- **Transformer Attention** adds ability to focus on **important motion events**.
- **Tabular MLP** adds non-PIR clues like ambient conditions.
- **Fusion** makes the model very **robust** to different sensor behaviors, environment conditions, and user activities.

---

# 📊 **Evaluation Strategy**
- **5-Fold Stratified K-Fold Cross Validation**: Ensures balanced testing across multiple runs.
- **Metrics Tracked:** Accuracy, Macro F1-Score, Class-wise reports, Confusion Matrices.

---

# 🖼️ **Diagram Sketch (high-level)**

```
    PIR Input → CNN → Flatten → 
                       \
    PIR Input → LSTM → LSTM → Transformer Attention → Flatten
                       /
    Tabular Input → Dense → 

(Merge CNN + LSTM + Attention + Tabular) → Dense → Dense → Output
```






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






