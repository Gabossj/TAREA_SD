# Testing of the Model
import pandas as pd
import numpy as np
from utility import *


def forward_softmax(X, Y, W, Vr):
    # Aplicar autoencoders
    for i in range(1, numAE + 1):
        X = act_sigmoid(np.dot(Vr[i], X))
    
    # Normalizar y aplicar softmax
    xv = zscores_dataset(X)
    zv = softmax(np.dot(W, xv))
    
    return zv


def measures(zv, Y):
    # Calcular matriz de confusión
    confusion = np.zeros((2, 2))
    
    # Obtener predicciones (clase con mayor probabilidad)
    predictions = np.argmax(zv, axis=0)
    true_labels = np.argmax(Y, axis=0)
    
    # Calcular TP, TN, FP, FN
    TP = np.sum((predictions == 0) & (true_labels == 0))
    TN = np.sum((predictions == 1) & (true_labels == 1))
    FP = np.sum((predictions == 0) & (true_labels == 1))
    FN = np.sum((predictions == 1) & (true_labels == 0))
    
    confusion[0, 0] = TP
    confusion[0, 1] = FP
    confusion[1, 0] = FN
    confusion[1, 1] = TN
    
    # Calcular métricas
    P = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
    R = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    F = 2 * (P * R) / (P + R) if (P + R) > 0 else 0  # F-score
    Acc = (TP + TN) / (TP + FP + TN + FN)  # Accuracy
    
    # Calcular clase predicha y F-scores por clase
    Clase1 = true_labels[0] if len(true_labels) > 0 else None
    Clase2 = true_labels[1] if len(true_labels) > 1 else None
    Promedios_Fscores = F
    
    return confusion, P, R, F, Acc, Clase1, Clase2, Promedios_Fscores


# Beginning ...
def main():
    X, Y = load_data()
    W = load_Wsoftmax()
    Vr = load_Vr()  # Cargar pesos de autoencoders
    
    X = zscores_dataset(X)
    
    zv = forward_softmax(X, Y, W, Vr)
    
    confusion, precision, recall, fscore, accuracy, clase1, clase2, promedio_fscores = measures(zv, Y)
    
    save_measures(confusion, precision, recall, fscore, accuracy, clase1, clase2, promedio_fscores)


if __name__ == '__main__':
    main()
