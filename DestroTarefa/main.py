import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler
from statistics import mean, stdev

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
scale = MaxAbsScaler().fit(arquivo[1])
y = np.ravel(scale.transform(arquivo[1]))
erroMaior = 100000

Erros = []
for vez in range(0,10):
    iter = 30000

    regr = MLPRegressor(hidden_layer_sizes=(15,15,15,15,15,15,15,15,15,15,15,15,15,15,15),
                        max_iter=iter,
                        activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=30000)
    print('Treinando RNA')
    regr = regr.fit(x,y)

    print('Preditor')
    y_est = regr.predict(x)

    if(regr.loss_curve_[-1] < erroMaior):
        erroMaior = regr.loss_curve_[-1]
        plt.figure(figsize=[14,7])
        #plot curso original
        plt.subplot(1,3,1)
        plt.plot(x,y)

        #plot aprendizagem
        plt.subplot(1,3,2)
        plt.plot(regr.loss_curve_)

        #plot regressor
        plt.subplot(1,3,3)
        plt.plot(x,y,linewidth=1,color='yellow')
        plt.plot(x,y_est,linewidth=2)

        plt.show()
    Erros.append(regr.loss_curve_[-1])

print(Erros)
print(np.average(Erros))
print(np.std(Erros))    