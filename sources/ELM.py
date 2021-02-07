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

def implement_shift_ELM(k,x_Treino):
    '''
    implementa shift a ser aplicado no vetor
    k: valor a ser shiftado
    X_Treino: vetor de treinamento
    
    '''
    
    #     3  i-3          2 i-2          1 i-1                  i
    all_vector = []    
    for i in range(k):        
        all_vector.append(x_Treino[i:i-k].values[::-1])
    X = np.c_[f(*all_vector)] 
    y = x_Treino['Monthly Mean Total Sunspot Number'][k:].values
                                                # i
    #x_Treino[3:3-k],x_Treino[2:2-k],x_Treino[1:1-k],x_Treino[:-k],np.ones(x_Treino[:-k].shape)
    
    
     
    return X,y

def f(*a):      
    return a


if __name__ == '__main__':

    df = pd.read_csv('monthly-sunspots.csv')
    # PROCESSING THE DATASET
    df = df.drop('Unnamed: 0',axis = 1)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date',inplace=True)

    # DIVIDING DATASET IN TRAIN,TEST
    x_Treino = df[df.index < pd.Timestamp('2010-01-01')]
    x_Teste = df[df.index >= pd.Timestamp('2010-01-01')]


    X_t = scipy.linalg.toeplitz(x_Treino[:8][::-1],x_Treino[7:-1])
    X,y_t = implement_shift_ELM(8,x_Treino) # < just to obtain the y_t value
    rmse_vector = []


    #### discovering the best number of neurons in the ELM layer
    for t in range(1,101):
        ## aplicando transformacao pela rede neural
        print(' Valor de T: {0}'.format(t))
        W = np.random.uniform(0,0.003,[t,8])
        x_new_t = np.tanh(W.dot(X_t)).transpose()
        
        kf = KFold(5)
        rmse_sum = 0
        rmse_len = 5
        for k,(train_index, test_index) in enumerate(kf.split(x_new_t,y_t)):
            clf = Ridge(alpha=0)
            clf.fit(x_new_t[train_index], y_t[train_index])
            rmse = sqrt(mean_squared_error(y_t[test_index], np.array(clf.predict(x_new_t[test_index]))))
            print("[fold {0}] , score: {1:.5f}, RMSE: {2}, Parametros:{3},{4}".
                format(k, clf.score(x_new_t[test_index], y_t[test_index]),rmse,clf.coef_,clf.intercept_))
            rmse_sum += rmse   
        print("Valor medio do RMSE: {0}".format(rmse_sum/rmse_len))
        rmse_vector.append(rmse_sum/rmse_len)


    
    ## FINDING THE BEST VALUE OF THE ALPHA PARAMETER, 
    resume_of_t = {}
    for t in range(1,101):
        ## aplicando transformacao pela rede neural
        print(' Valor de T: {0}'.format(t))
        W = np.random.uniform(0,0.003,[t,8])
        x_new_t = np.tanh(W.dot(X_t)).transpose()
        
        kf = KFold(5)    
        a_params = [2**-12,2**-11,2**-10,2**-9,2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**10,2**11,2**12] 
        alpha_results = []
        rmse_vector = []
        resume_of_rounds = {}
        for a in a_params:
            rmse_sum = 0
            rmse_len = 5
            for k,(train_index, test_index) in enumerate(kf.split(x_new_t,y_t)):
                clf = Ridge(alpha=a)
                clf.fit(x_new_t[train_index], y_t[train_index])
                rmse = sqrt(mean_squared_error(y_t[test_index], np.array(clf.predict(x_new_t[test_index]))))
                rmse_sum += rmse         
            print("Valor medio do RMSE: {0}\n".format(rmse_sum/rmse_len))
            print('Melhor valor de alpha:{0}\n'.format(a))
            ## X_new é o novo vetor a ser implementado o metodo de mmq
            resume_of_rounds[a] = rmse_sum/rmse_len
        resume_of_t[t] = resume_of_rounds

    
    # performing o test set
    X_t,y_t = implement_shift_2(8,x_Treino)
    X_t = X_t.transpose()

    X_t = scipy.linalg.toeplitz(x_Treino[:8][::-1],x_Treino[7:-1])
    X,y_t = implement_shift_2(8,x_Treino)


    W = np.random.uniform(0,0.003,[98,8])
    x_new_t = np.tanh(W.dot(X_t)).transpose()
    clf = Ridge(alpha=2**-5)
    clf.fit(x_new_t, y_t)

    X_teste = scipy.linalg.toeplitz(x_Teste[:8][::-1],x_Teste[7:-1])
    X,y_teste = implement_shift_2(8,x_Teste)
    x_new_teste = np.tanh(W.dot(X_teste)).transpose()
    rmse = sqrt(mean_squared_error(y_teste, np.array(clf.predict(x_new_teste))))