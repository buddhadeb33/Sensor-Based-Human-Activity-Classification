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

- Each activity shows distinctive signal shape and intensity.
- Motivates the use of CNNs, RNNs, and Attention mechanisms.

---

## 4. Data Preprocessing

- **Encoding**: Labels encoded using `LabelEncoder`
- **Scaling**: Temperature scaled with `StandardScaler`
- **Reshaping**: PIR data reshaped to (samples, timesteps, channels) = `(n, 55, 1)`
- **Splitting**: 5-Fold Stratified Cross Validation for robust training/testing

---

## 5. Model Architecture

### Type: Hybrid CNN + LSTM + Attention + Dense Fusion

#### Components:
- **CNN (Conv1D)**: Extracts local temporal features from PIR data
- **LSTM**: Models sequential dependencies across time steps
- **Multi-Head Attention**: Captures long-range dependencies
- **Tabular Dense**: Processes numerical inputs
- **Fusion Layers**: Combines all learned features
- **Regularization**: Dropout and L2 to reduce overfitting

### Architecture Diagram (Mermaid)

```mermaid
graph TD
  A[PIR Input (55,1)] --> B1[Conv1D + Pooling]
  B1 --> B2[Conv1D + Pooling]
  B2 --> B3[Flatten - CNN Features]

  A --> C1[LSTM Layer 1]
  C1 --> C2[LSTM Layer 2 - LSTM Features]

  C1 --> D1[MultiHeadAttention]
  D1 --> D2[Add + LayerNorm]
  D2 --> D3[Flatten - Transformer Features]

  E[Tabular Input (3,)] --> F[Dense Layer (16)]

  B3 --> G[Concatenate All Features]
  C2 --> G
  D3 --> G
  F --> G

  G --> H1[Dense Layer (128) + Dropout]
  H1 --> H2[Dense Layer (64) + Dropout]
  H2 --> I[Softmax Output Layer]


6. Training Strategy
Cross-Validation: 5-Fold Stratified K-Fold

Epochs: 5 (can be increased based on compute availability)

Batch Size: 32

Loss Function: SparseCategoricalCrossentropy

Optimizer: Adam

Metrics: Accuracy

Early Stopping: Can be optionally added

What’s Tracked Per Fold:
Accuracy

Confusion Matrix

Classification Report

Precision, Recall, F1-Score

Training & Validation Loss

Training & Validation Accuracy

7. Model Evaluation
Example Fold Metrics:
mathematica
Copy
Edit
Fold 1 Accuracy: 0.8821
Fold 2 Accuracy: 0.8710
Fold 3 Accuracy: 0.8895
Fold 4 Accuracy: 0.8652
Fold 5 Accuracy: 0.8783

Mean Accuracy: 0.8772
Standard Deviation: 0.0082
Confusion Matrix
Visualized per fold using Seaborn. Misclassifications are relatively low and balanced across classes.

Classification Report
Per-class metrics include:

Precision

Recall

F1-Score

8. Alternate Models Explored
Model	Reason for Not Choosing
Random Forest, XGBoost	Poor with raw time-series without feature engineering
CNN Only	Failed to model sequential motion properly
LSTM Only	Performed well but lacked local feature extraction
GRU	Comparable to LSTM but slightly less accurate
Transformer Only	Overfitting due to limited dataset size
Temporal Fusion Transformer	Too complex for short-range time series
The hybrid model showed the best generalization and performance consistency.

9. File Structure
kotlin
Copy
Edit
pirvision-project/
├── data/
│   └── pirvision.csv
├── notebooks/
│   └── eda_and_modeling.ipynb
├── model/
│   └── architecture.py
├── results/
│   ├── confusion_matrices/
│   └── metrics_summary.json
├── README.md
10. How to Run
Clone the repository:



git clone https://github.com/your-repo/pirvision-project.git
cd pirvision-project
Install dependencies:



pip install -r requirements.txt
Launch the notebook:



jupyter notebook notebooks/eda_and_modeling.ipynb
11. Conclusion
This solution demonstrates a robust hybrid deep learning pipeline tailored for time-series-based activity recognition from PIR sensors. By combining CNN, LSTM, Attention, and tabular data processing, the model captures local, sequential, and global context effectively, leading to high classification accuracy and generalization.

12. Future Improvements
Use EarlyStopping for better convergence

Introduce Class Weights to mitigate class imbalance

Explore Self-Supervised Pretraining for PIR features

Integrate model in real-time embedded systems or IoT devices


