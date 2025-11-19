import pandas as pd
import numpy as np
import os
from utility import *

def updW_adam(W, V, S, gW, mu, t):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    V = beta1 * V + (1 - beta1) * gW
    S = beta2 * S + (1 - beta2) * (gW ** 2)
    factor = np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    W = W - mu * factor * V / (np.sqrt(S) + eps)
    return W, V, S

def softmax_grad(Act, xe, ye):
    M = xe.shape[1]
    error = Act - ye
    gW = error @ xe.T / M
    Cost = -np.sum(ye * np.log(Act + 1e-9)) / M
    return gW, Cost

def calcula_Batch(N, BatchSize):
    return int(np.ceil(N / BatchSize))

def get_miniBatch(n, X, Y, BatchSize):
    start_idx = (n - 1) * BatchSize
    end_idx = min(n * BatchSize, X.shape[1])
    return X[:, start_idx:end_idx], Y[:, start_idx:end_idx]

def train_miniBatch(Xe, Ye, W, V, S, mu, BatchSize, iter_num):
    NumBatch = calcula_Batch(Xe.shape[1], BatchSize)
    Cost = np.zeros(NumBatch)
    
    for n in range(1, NumBatch + 1):
        xe, ye = get_miniBatch(n, Xe, Ye, BatchSize)
        Act = softmax(W @ xe)
        gW, Cost[n-1] = softmax_grad(Act, xe, ye)
        t = iter_num * NumBatch + n
        W, V, S = updW_adam(W, V, S, gW, mu, t)
    
    return np.mean(Cost), W, V, S

def evaluate_accuracy(X, Y, W):
    predictions = softmax(W @ X)
    y_pred = np.argmax(predictions, axis=0)
    y_true = np.argmax(Y, axis=0)
    return np.mean(y_pred == y_true)

def train_softmax(X, Y, param):
    Nprev = X.shape[0]
    Nclass = Y.shape[0]
    
    W = iniW(Nclass, Nprev) * 0.01
    V = np.zeros_like(W)
    S = np.zeros_like(W)
    
    best_acc = 0
    best_W = W.copy()
    patience = 20
    patience_counter = 0
    
    for Iter in range(1, param['MaxIter'] + 1):
        idx = np.random.permutation(X.shape[1])
        Xe = X[:, idx]
        Ye = Y[:, idx]
        
        Cost, W, V, S = train_miniBatch(Xe, Ye, W, V, S, param['mu'], param['BatchSize'], Iter)
        
        if Iter % 50 == 0:
            acc = evaluate_accuracy(X, Y, W)
            print(f"Iter {Iter}/{param['MaxIter']}: Cost={Cost:.6f}, Acc={acc*100:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_W = W.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at iteration {Iter}")
                break
    
    W = best_W
    final_acc = evaluate_accuracy(X, Y, W)
    print(f"Best Acc: {best_acc*100:.2f}%, Final Acc: {final_acc*100:.2f}%")
    return W, [Cost]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'DATA')
    
    config = pd.read_csv(os.path.join(script_dir, 'config_softmax.csv'), header=None).values.flatten()
    X = pd.read_csv(os.path.join(script_dir, 'Xtrain_reduced.csv'), header=None).values.T
    Y = pd.read_csv(os.path.join(data_path, 'classtrain.csv'), header=None).values.T
    
    param = {
        'MaxIter': int(config[0]),
        'mu': float(config[1]),
        'BatchSize': int(config[2])
    }
    
    print(f"Softmax: {X.shape[0]} features -> {Y.shape[0]} classes, MaxIter={param['MaxIter']}, mu={param['mu']}, BatchSize={param['BatchSize']}\n")
    
    W, Cost = train_softmax(X, Y, param) 
    
    np.save(os.path.join(script_dir, 'W_softmax.npy'), W)
    pd.DataFrame({'Cost': Cost}).to_csv(os.path.join(script_dir, 'cost_history.csv'), index=False)
    
    print(f"\nFinal Cost: {Cost[-1]:.6f}")

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nExecution time: {elapsed:.2f}s")
    print("✓ F2 MET" if elapsed <= 60 else "✗ F2 NOT MET")