import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datetime import datetime,date
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import scipy


def implement_shift(k,x_Treino):
    '''
    implementa shift a ser aplicado no vetor
    k: valor a ser shiftado
    X_Treino: vetor de treinamento
    
    '''
    all_vector = []    
    for i in range(k):        
        all_vector.append(x_Treino[i:i-k].values)
    all_vector.append(np.ones(x_Treino[:-k].shape))
    X = np.c_[f(*all_vector)] 
    y = x_Treino['Monthly Mean Total Sunspot Number'][k:].values
    
    return X,y
def f(*a):
    '''
    retrieves a different format in vector
    '''
    return a

def implementa_regressao(n,x_Treino):
    '''
    n : n-fold
    x_Treino: vetor a ser autoregredido
    '''
    kf = KFold(5)
    X,y  = implement_shift(n,x_Treino)
    rmse_sum = 0
    rmse_len = 5
    for k,(train_index, test_index) in enumerate(kf.split(X,y)):
        reg = linear_model.LinearRegression().fit(X[train_index], y[train_index])
        rmse = sqrt(mean_squared_error(y[test_index], np.array(reg.predict(X[test_index]))))
        print("[fold {0}] , score: {1:.5f}, RMSE: {2}, Parametros:{3},{4}".
              format(k, reg.score(X[test_index], y[test_index]),rmse,reg.coef_,reg.intercept_))
        rmse_sum += rmse
    print("Valor medio do RMSE: {0}".format(rmse_sum/rmse_len))
    
    return rmse_sum/rmse_len


if __name__ == '__main__':
    df = pd.read_csv('monthly-sunspots.csv')
    # PROCESSING THE DATASET
    df = df.drop('Unnamed: 0',axis = 1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date',inplace=True)

    # DIVIDING DATASET IN TRAIN,TEST
    x_Treino = df[df.index < pd.Timestamp('2010-01-01')]
    x_Teste = df[df.index >= pd.Timestamp('2010-01-01')]


    # finding the best RMSE value for the delays considered between 1 and 25
    rmse = []
    for i in range(1,25):
        rmse.append(implementa_regressao(i,x_Treino))


    ### TEST SET
    X,y = implement_shift(24,x_Teste)
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    y_predict = X.dot(W)

    # FINAL VALUE OF RMSE
    rmse = sqrt(mean_squared_error(y, np.array(y_predict)))