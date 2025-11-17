# Testing of the Model - Improved Version
import pandas as pd
import numpy as np
import os
from utility import *

def forward_softmax(X, W):
    return softmax(W @ X)

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    confusion_matrix = np.array([[tp, fp], 
                                 [fn, tn]])
    
    precision_class1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision_class2 = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    recall_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_class2 = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1) if (precision_class1 + recall_class1) > 0 else 0
    f1_class2 = 2 * (precision_class2 * recall_class2) / (precision_class2 + recall_class2) if (precision_class2 + recall_class2) > 0 else 0
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    fscore_avg = (f1_class1 + f1_class2) / 2
    
    fscores = np.array([f1_class1, f1_class2, fscore_avg])
    
    return confusion_matrix, fscores, accuracy

def measures(y_true, y_pred):
    confusion_matrix, fscores, accuracy = calculate_metrics(y_true, y_pred)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    confusion_df = pd.DataFrame(confusion_matrix, 
                                columns=['Positivo_Pred', 'Negativo_Pred'], 
                                index=['Positivo_Real', 'Negativo_Real'])
    
    confusion_path = os.path.join(script_dir, 'confusión.csv')
    confusion_df.to_csv(confusion_path, index=True)
    
    fscores_df = pd.DataFrame(fscores.reshape(1, 3), 
                              columns=['Clase1', 'Clase2', 'Promedio'])
    
    fscores_path = os.path.join(script_dir, 'fscores.csv')
    fscores_df.to_csv(fscores_path, index=False)
    
    return confusion_matrix, fscores, accuracy

def normalize_with_stats(X, mean, std):
    """Aplicar normalización con estadísticas guardadas"""
    return (X - mean) / std

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    # Verificar archivos necesarios
    required_files = {
        'W_softmax.npy': os.path.join(script_dir, 'W_softmax.npy'),
        'model_rdim.npz': os.path.join(script_dir, 'model_rdim.npz'),
        'dtest.csv': os.path.join(data_path, 'dtest.csv'),
        'classtest.csv': os.path.join(data_path, 'classtest.csv')
    }
    
    for file_name, file_path in required_files.items():
        if not os.path.exists(file_path):
            print(f"ERROR: No se encuentra el archivo {file_name} en {file_path}")
            print("Asegúrate de haber ejecutado primero:")
            print("1. rdim_mejorado.py (para generar model_rdim.npz)")
            print("2. trn_mejorado.py (para generar W_softmax.npy)")
            return
    
    # Leer datos de TEST (NO de train)
    X = pd.read_csv(required_files['dtest.csv'], header=None).values.T
    Y = pd.read_csv(required_files['classtest.csv'], header=None).values.T
    
    # Cargar pesos y modelo
    W = np.load(required_files['W_softmax.npy'])
    model = np.load(required_files['model_rdim.npz'], allow_pickle=True)
    V1 = model['V1']
    V2 = model['V2']
    V3 = model['V3']
    
    # Cargar estadísticas de normalización
    means = model['means']
    stds = model['stds']
    
    print(f"{'='*50}")
    print("TESTING PIPELINE")
    print(f"{'='*50}")
    print(f"Test data shape: {X.shape}")
    print(f"Test labels shape: {Y.shape}")
    print(f"Softmax weights shape: {W.shape}")
    print(f"Model V1: {V1.shape}, V2: {V2.shape}, V3: {V3.shape}")
    
    # Aplicar el mismo pipeline que en entrenamiento
    print(f"\n{'='*50}")
    print("APPLYING TRANSFORMATIONS")
    print(f"{'='*50}")
    
    # 1. Normalización inicial
    X = normalize_with_stats(X, means[0], stds[0])
    print(f"After initial normalization: {X.shape}")
    
    # 2. Primera capa autoencoder
    X = act_sigmoid(V1 @ X)
    X = normalize_with_stats(X, means[1], stds[1])
    print(f"After V1 + normalization: {X.shape}")
    
    # 3. Segunda capa autoencoder
    X = act_sigmoid(V2 @ X)
    X = normalize_with_stats(X, means[2], stds[2])
    print(f"After V2 + normalization: {X.shape}")
    
    # 4. PCA
    X = V3 @ X
    X = normalize_with_stats(X, means[3], stds[3])
    print(f"After V3 + normalization: {X.shape}")
    
    # 5. Aplicar softmax y obtener predicciones
    zv = forward_softmax(X, W)
    y_pred = np.argmax(zv, axis=0)
    y_true = np.argmax(Y, axis=0)
    
    # Calcular y mostrar métricas
    confusion_matrix, fscores, accuracy = measures(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print("RESULTADOS DEL TESTING")
    print(f"{'='*50}")
    
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(f"Verdaderos Positivos (TP): {confusion_matrix[0,0]}")
    print(f"Falsos Positivos (FP): {confusion_matrix[0,1]}")
    print(f"Falsos Negativos (FN): {confusion_matrix[1,0]}")
    print(f"Verdaderos Negativos (TN): {confusion_matrix[1,1]}")
    
    print("\n--- F-SCORES ---")
    print(f"Clase 1 (F1-score): {fscores[0]:.4f}")
    print(f"Clase 2 (F1-score): {fscores[1]:.4f}")
    print(f"Promedio (F1-score): {fscores[2]:.4f}")
    
    print(f"\n--- EXACTITUD (ACCURACY) ---")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n{'='*50}")
    if accuracy >= 0.95:
        print("OBJETIVO ALCANZADO: Accuracy >= 95%")
    else:
        print(f"Objetivo no alcanzado. Faltan {(0.95-accuracy)*100:.2f} puntos porcentuales")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()