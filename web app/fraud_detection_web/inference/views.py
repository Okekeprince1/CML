import os
import time
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.conf import settings
import boto3
from tensorflow import keras

INFERENCE_DIR = os.path.join(os.path.dirname(__file__))

MODEL_FILES = {
    'logistic_regression': os.path.join('models', 'logistic_regression_model.pkl'),
    'neural_network': os.path.join('models', 'neural_network_model.h5'),
    'knn': os.path.join('models', 'knn_model.pkl'),
    'decision_tree': os.path.join('models', 'decision_tree_model.pkl'),
}

PREPROCESSOR_FILE = os.path.join('models', 'preprocessor.pkl')
X_TEST_FILE = 'X_test_processed.csv'
Y_TEST_FILE = 'y_test_processed.csv'

MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'neural_network': 'Neural Network',
    'knn': 'KNN',
    'decision_tree': 'Decision Tree',
}

S3_BUCKET = 'cml-output'
S3_FILES = {
    'X_test_processed.csv': 'X_test_processed.csv',
    'y_test_processed.csv': 'y_test_processed.csv',
}

def download_from_s3_if_not_exists():
    try:
        s3 = boto3.client('s3')
        for s3_key, local_file in S3_FILES.items():
            local_path = os.path.join(INFERENCE_DIR, local_file)
            if not os.path.exists(local_path):
                print(f"Downloading {s3_key} from S3 to {local_path}")
                s3.download_file(S3_BUCKET, s3_key, local_path)
        else:
            print(f"{local_file} already exists")
    except Exception as e:
        print(f"Error downloading files from S3: {e}")

def load_model(model_key):
    path = os.path.join(INFERENCE_DIR, MODEL_FILES[model_key])
    if model_key == 'neural_network':
        model = keras.models.load_model(path)
        metadata = joblib.load(path.replace('.h5', '_metadata.pkl'))
        config = metadata['config']
        training_time = metadata['training_time']
    else:
        model_data = joblib.load(path)
        model = model_data['model']
        config = model_data['config']
        training_time = model_data['training_time']
    print(f"loaded model with details: config-{config}, training time-{training_time}")
    return model


download_from_s3_if_not_exists()

X_test_file = pd.read_csv(os.path.join(INFERENCE_DIR, X_TEST_FILE))
y_test_file = pd.read_csv(os.path.join(INFERENCE_DIR, Y_TEST_FILE)).iloc[:, 0]

sample_size = min(100, len(X_test_file))
X_test = X_test_file.sample(n=sample_size, random_state=42)
y_test = y_test_file.iloc[X_test.index].sample(n=sample_size, random_state=42)

def model_inference(request):
    context = {}
    if request.method == 'POST':
        model_key = request.POST.get('model')
        if model_key in MODEL_FILES:
            model = load_model(model_key)
            X_proc = X_test
            start = time.time()
            if model_key == 'neural_network':
                y_pred = (model.predict(X_proc) > 0.5).astype(int).flatten()
            else:
                y_pred = model.predict(X_proc)
            inference_time = time.time() - start
            actual = y_test[:100].tolist()
            print("actual", actual)
            predicted = y_pred[:100].tolist()
            print("predicted", predicted)
            chart_labels = list(range(1, len(actual)+1))
            chart_data = [{'x': a, 'y': p} for a, p in zip(actual, predicted)]
            context = {
                'predictions': True,
                'inference_time': f"{inference_time:.2f}",
                'actual': actual,
                'predicted': predicted,
                'chart_labels': chart_labels,
                'chart_data': chart_data,
                'model_key': model_key,
                'model': MODEL_LABELS[model_key],
            }
    return render(request, 'inference/model_inference.html', context)

def model_comparison(request):
    context = {}
    if request.method == 'POST':
        model1_key = request.POST.get('model1')
        model2_key = request.POST.get('model2')
        if model1_key in MODEL_FILES and model2_key in MODEL_FILES:
            model1 = load_model(model1_key)
            model2 = load_model(model2_key)
            X_proc = X_test
            
            start1 = time.time()
            if model1_key == 'neural_network':
                y_pred1 = (model1.predict(X_proc) > 0.5).astype(int).flatten()
            else:
                y_pred1 = model1.predict(X_proc)
            inference_time1 = time.time() - start1
            start2 = time.time()
            if model2_key == 'neural_network':
                y_pred2 = (model2.predict(X_proc) > 0.5).astype(int).flatten()
            else:
                y_pred2 = model2.predict(X_proc)
            inference_time2 = time.time() - start2
            accuracy1 = np.mean(y_pred1 == y_test)
            accuracy2 = np.mean(y_pred2 == y_test)
            difference = accuracy1 - accuracy2
            
            chart_labels = list(range(1, len(y_pred1) + 1))
            model1_data = y_pred1.tolist()
            model2_data = y_pred2.tolist()
            
            context = {
                'comparison': {
                    'accuracy1': f"{accuracy1:.3f}",
                    'accuracy2': f"{accuracy2:.3f}",
                    'difference': f"{difference:.3f}",
                    'inference_time1': f"{inference_time1:.2f}",
                    'inference_time2': f"{inference_time2:.2f}",
                    'chart_labels': chart_labels,
                    'model1_data': model1_data,
                    'model2_data': model2_data,
                },
                'model1': MODEL_LABELS[model1_key],
                'model2': MODEL_LABELS[model2_key],
            }
    return render(request, 'inference/model_comparison.html', context)