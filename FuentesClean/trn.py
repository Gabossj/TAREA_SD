# Training Softmax via mAdam algorithm
import pandas as pd
import numpy as np
from utility import *


def updW_adam(W, V, S, gW, mu, n):
    V = mu * V + (1 - mu) * gW
    S = mu * S + (1 - mu) * (gW ** 2)
    V_hat = V / (1 - mu ** n)
    S_hat = S / (1 - mu ** n)
    W = W - (eta * V_hat) / (np.sqrt(S_hat) + epsilon)
    return W, V, S


def softmax_grad(Act, xe, ye):
    gW = np.dot((Act - ye), xe.T)
    return gW


def softmax(xe, W):
    z = np.dot(W, xe)
    exp_z = np.exp(z - np.max(z))
    Act = exp_z / np.sum(exp_z)
    return Act


def train_minibatch(Xe, Ye, param):
    NumBatch = calcula_Batch()
    Cost = []
    for n in range(1, NumBatch + 1):
        xe, ye = get_miniBtach(n, Xe, Ye, BtachSize)
        Act = softmax(xe, W)
        gW, Cost_n = softmax_grad(Act, xe, ye)
        W, V, S = updW_adam(W, V, S, gW, mu, n)
        Cost.append(Cost_n)
    
    CostAvg = np.mean(Cost)
    return CostAvg, W, V, S


def train_softmax():
    X, Y = load_data()
    X = zcores_dataset(X)
    W = randW()
    V, S = zeros()
    
    for Iter in range(1, MaxIter + 1):
        Xe, Ye = rand_position_data(X, Y)
        W, V, S, Cost[Iter] = train_minibatch(Xe, Ye, W, V, S, mu, BatchSize)
        if Iter % 100 == 0:
            print(f"Cost({Iter})")
    
    Save_W_Costo(W, Cost)


def main():
    load_data()
    zcores_data()
    train_softmax()
    save_W_Costo()


if __name__ == '__main__':
    main()
