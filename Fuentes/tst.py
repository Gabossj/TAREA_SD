# Testing of the Model
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
    
    return confusion_matrix, fscores

def measures(y_true, y_pred):
    confusion_matrix, fscores = calculate_metrics(y_true, y_pred)
    
    # Obtener la ruta del directorio del script para guardar archivos
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
    
    return confusion_matrix, fscores

def main():
    # Obtener la ruta del directorio donde está el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ruta a la carpeta DATA
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    # Verificar que existen los archivos necesarios
    required_files = {
        'config_sae.csv': os.path.join(script_dir, 'config_sae.csv'),
        'W_softmax.npy': os.path.join(script_dir, 'W_softmax.npy'),
        'model_rdim.npz': os.path.join(script_dir, 'model_rdim.npz'),
        'dtrain.csv': os.path.join(data_path, 'dtrain.csv'),
        'classtest.csv': os.path.join(data_path, 'classtest.csv')
    }
    
    # Verificar existencia de archivos
    for file_name, file_path in required_files.items():
        if not os.path.exists(file_path):
            print(f"ERROR: No se encuentra el archivo {file_name} en {file_path}")
            print("Asegúrate de haber ejecutado primero:")
            print("1. rdim.py (para generar model_rdim.npz)")
            print("2. trn.py (para generar W_softmax.npy)")
            return
    
    # Leer config_sae.csv
    config = pd.read_csv(required_files['config_sae.csv'], header=None).values.flatten()
    
    # Leer dtrain.csv y classtest.csv
    X = pd.read_csv(required_files['dtrain.csv'], header=None).values.T
    Y = pd.read_csv(required_files['classtest.csv'], header=None).values.T
    
    # Cargar pesos y modelo usando rutas absolutas
    W = np.load(required_files['W_softmax.npy'])
    model = np.load(required_files['model_rdim.npz'])
    V1 = model['V1']
    V2 = model['V2']
    V3 = model['V3']
    
    print(f"Datos de entrada: {X.shape}")
    print(f"Etiquetas: {Y.shape}")
    print(f"Pesos Softmax: {W.shape}")
    print(f"Modelo V1: {V1.shape}, V2: {V2.shape}, V3: {V3.shape}")
    
    # Normalizar datos y aplicar autoencoders
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    X = act_sigmoid(V1 @ X)
    print(f"Después de V1: {X.shape}")
    X = act_sigmoid(V2 @ X)
    print(f"Después de V2: {X.shape}")
    X = V3 @ X
    print(f"Después de V3: {X.shape}")
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    
    # Aplicar softmax y obtener predicciones
    zv = forward_softmax(X, W)
    y_pred = np.argmax(zv, axis=0)
    y_true = np.argmax(Y, axis=0)
    
    print(f"Predicciones: {y_pred.shape}")
    print(f"Etiquetas reales: {y_true.shape}")
    
    # Calcular y mostrar métricas
    confusion_matrix, fscores = measures(y_true, y_pred)
    
    print("\n" + "="*50)
    print("RESULTADOS DEL TESTING")
    print("="*50)
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(f"Verdaderos Positivos (TP): {confusion_matrix[0,0]}")
    print(f"Falsos Positivos (FP): {confusion_matrix[0,1]}")
    print(f"Falsos Negativos (FN): {confusion_matrix[1,0]}")
    print(f"Verdaderos Negativos (TN): {confusion_matrix[1,1]}")
    
    print("\n--- F-SCORES ---")
    print(f"Clase 1 (F1-score): {fscores[0]:.4f}")
    print(f"Clase 2 (F1-score): {fscores[1]:.4f}")
    print(f"Promedio (F1-score): {fscores[2]:.4f}")
    
    # Calcular accuracy adicional
    accuracy = np.mean(y_true == y_pred)
    print(f"\n--- EXACTITUD (ACCURACY) ---")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()