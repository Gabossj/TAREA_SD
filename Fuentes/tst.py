import pandas as pd
import numpy as np
import os
from utility import *

def forward_softmax(X, W):
    return softmax(W @ X)   

def calculate_multiclass_metrics(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_class, pred_class in zip(y_true, y_pred):
        confusion_matrix[true_class, pred_class] += 1 
    
    fscores = []
    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fscores.append(f1)
    
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    fscore_avg = np.mean(fscores)
    
    return confusion_matrix, np.array(fscores), fscore_avg, accuracy

def apply_reduction_pipeline(X, model):
    V1 = model['V1']
    V2 = model['V2']
    V3 = model['V3']
    X_mean = model['X_mean']
    X_std = model['X_std']
    svd_mean = model['svd_mean']
    
    X = (X - X_mean) / X_std
    X = act_sigmoid(V1 @ X)
    X = act_sigmoid(V2 @ X)
    X = V3 @ (X - svd_mean)
    return X

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    X = pd.read_csv(os.path.join(data_path, 'dtest.csv'), header=None).values.T
    Y = pd.read_csv(os.path.join(data_path, 'classtest.csv'), header=None).values.T
    W = np.load(os.path.join(script_dir, 'W_softmax.npy'))
    model = np.load(os.path.join(script_dir, 'model_rdim.npz'), allow_pickle=True)
    
    X_reduced = apply_reduction_pipeline(X, model)
    
    predictions = forward_softmax(X_reduced, W)
    y_pred = np.argmax(predictions, axis=0)
    y_true = np.argmax(Y, axis=0)
    
    num_classes = Y.shape[0]
    confusion_matrix, fscores, fscore_avg, accuracy = calculate_multiclass_metrics(y_true, y_pred, num_classes)
    
    print("\nCONFUSION MATRIX:")
    print(confusion_matrix)
    
    # Mostrar matriz de confusión detallada para caso binario
    if num_classes == 2:
        print("\n--- MATRIZ DE CONFUSIÓN DETALLADA ---")
        print(f"Verdaderos Positivos (TP): {confusion_matrix[0,0]}")
        print(f"Falsos Positivos (FP): {confusion_matrix[0,1]}")
        print(f"Falsos Negativos (FN): {confusion_matrix[1,0]}")
        print(f"Verdaderos Negativos (TN): {confusion_matrix[1,1]}")
    
    print("\nF-SCORES:")
    for i, fscore in enumerate(fscores):
        print(f"Class {i}: {fscore:.4f} ({fscore*100:.2f}%)")
    
    print(f"\nMean F-score: {fscore_avg:.4f} ({fscore_avg*100:.2f}%)")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n{'✓ F5 MET' if fscore_avg >= 0.90 else '✗ F5 NOT MET'}")
    
    pd.DataFrame(confusion_matrix).to_csv(os.path.join(script_dir, 'confusion.csv'), index=False, header=False)
    fscores_df = pd.DataFrame({
        'Class': [f'Class_{i}' for i in range(num_classes)] + ['Average'],
        'F-Score': list(fscores) + [fscore_avg]
    })
    fscores_df.to_csv(os.path.join(script_dir, 'fscores.csv'), index=False)

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nExecution time: {elapsed:.2f}s")
    print("✓ F3 MET" if elapsed <= 5 else "✗ F3 NOT MET")