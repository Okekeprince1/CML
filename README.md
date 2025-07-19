# Fraud Detection Machine Learning Project

## Project Overview
This project implements a comprehensive fraud detection system using multiple machine learning techniques, focusing on balancing performance, latency/infrerence time. The solution addresses credit card fraud detection using existing train and test datasets.

## Dataset
- **Files**: 
  - `fraudTrain.csv` - Training dataset
  - `fraudTest.csv` - Test dataset

## Team Member Responsibilities

### Member 1: Logistic Regression
- Literature review on traditional ML for fraud detection
- Implementation of baseline model
- Performance evaluation and comparison

### Member 2: Neural Networks (Deep Learning)
- Literature review on deep learning for fraud detection
- Implementation of neural network models
- Advanced feature engineering

### Member 3: K-Nearest Neighbors (KNN)
- Literature review on KNN for fraud detection
- Implementation of KNN model
- Model evaluation and comparison

### Member 4: Decision Tree
- Literature review on decision tree methods
- Implementation of decision tree model
- Model evaluation and comparison

## Key Requirements Addressed

### 1. Data Collection and Transformation
- Loading existing train and test datasets
- Feature engineering and preprocessing
- Data validation and quality checks

### 2. Evaluation Framework
- Standard metrics: Precision, Recall, F1-Score, AUC-ROC

## Getting Started

### Prerequisites
1. Ensure you have the dataset files in the project root:
   - `fraudTrain.csv`
   - `fraudTest.csv`

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. **Run the complete pipeline:**
```bash
python src/main.py
```

2. **Run specific steps:**
```bash
# Generate Feature Peformance plots
python src/main.py --plot-only

```

## Data Processing Pipeline

### Step 1: Data Loading
- Loads existing `fraudTrain.csv` and `fraudTest.csv` files
- Validates data integrity and structure
- Separates features and target variables

### Step 2: Data Preprocessing
- Handles missing values and duplicates
- Performs feature engineering:
  - Log and square root transformations for Amount
  - Time-based features (hour, day)
  - Statistical aggregations for V features
- Scales numerical features using RobustScaler
- Handles class imbalance using undersampling

### Step 3: Model Training
- **Logistic Regression**: Traditional ML approach
- **Neural Networks**: Deep learning with TensorFlow/Keras
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **Decision Tree**: Tree-based classification

### Step 4: Model Evaluation
- Comprehensive performance metrics
- Model comparison and visualization
- Error analysis and insights

## Evaluation Criteria
- Model Performance: >70% accuracy target
- Comprehensive error analysis and documentation

## Expected Performance
- **Logistic Regression**: 85-90% accuracy
- **Neural Networks**: 92-95% accuracy
- **KNN**: 85-90% accuracy
- **Decision Tree**: 85-92% accuracy 