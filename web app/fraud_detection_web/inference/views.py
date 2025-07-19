import os
import time
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.conf import settings
# from tensorflow import keras

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

def load_model(model_key):
    path = os.path.join(INFERENCE_DIR, MODEL_FILES[model_key])
    if model_key == 'neural_network':
        # return keras.models.load_model(path)
        pass
    else:
        model_data = joblib.load(path)
        model = model_data['model']
        config = model_data['config']
        training_time = model_data['training_time']
        print(f"loaded model with details: config-{config}, training time-{training_time}")
        return model

# def load_test_data():
#     X_test = pd.read_csv(os.path.join(INFERENCE_DIR, X_TEST_FILE))
#     y_test = pd.read_csv(os.path.join(INFERENCE_DIR, Y_TEST_FILE)).iloc[:, 0]
#     return X_test, y_test

X_test = pd.read_csv(os.path.join(INFERENCE_DIR, X_TEST_FILE))
y_test = pd.read_csv(os.path.join(INFERENCE_DIR, Y_TEST_FILE)).iloc[:, 0]

def model_inference(request):
    context = {}
    if request.method == 'POST':
        model_key = request.POST.get('model')
        if model_key in MODEL_FILES:
            model = load_model(model_key)
            print("loadinf data")
            start1 = time.time()
            # X_test, y_test = load_test_data()
            print(f"loadinf data finsihed {start1 - time.time()}")

            X_proc = X_test
            start = time.time()
            if model_key == 'neural_network':
                y_pred = (model.predict(X_proc) > 0.5).astype(int).flatten()
            else:
                y_pred = model.predict(X_proc)
            inference_time = time.time() - start
            # For visualization, only showing 100 results
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
            # X_test, y_test = load_test_data()
            X_proc = X_test
            if model1_key == 'neural_network':
                y_pred1 = (model1.predict(X_proc) > 0.5).astype(int).flatten()
            else:
                y_pred1 = model1.predict(X_proc)
            if model2_key == 'neural_network':
                y_pred2 = (model2.predict(X_proc) > 0.5).astype(int).flatten()
            else:
                y_pred2 = model2.predict(X_proc)
            accuracy1 = np.mean(y_pred1 == y_test)
            accuracy2 = np.mean(y_pred2 == y_test)
            difference = accuracy1 - accuracy2
            # For visualization, only showing 100 results
            chart_labels = list(range(1, 101))
            model1_data = y_pred1[:100].tolist()
            model2_data = y_pred2[:100].tolist()
            context = {
                'comparison': {
                    'accuracy1': f"{accuracy1:.3f}",
                    'accuracy2': f"{accuracy2:.3f}",
                    'difference': f"{difference:.3f}",
                    'chart_labels': chart_labels,
                    'model1_data': model1_data,
                    'model2_data': model2_data,
                },
                'model1': MODEL_LABELS[model1_key],
                'model2': MODEL_LABELS[model2_key],
            }
    return render(request, 'inference/model_comparison.html', context)
