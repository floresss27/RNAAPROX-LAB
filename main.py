import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler
import os
import statistics

arquivos = ['teste2.npy', 'teste3.npy', 'teste4.npy', 'teste5.npy']
arquiteturas = [
    (10,),
    (20, 10),
    (50, 30, 10),
]
iteracoes = 1000
execucoes = 10

def executar_teste(x, y, arq):
    erros = []
    modelos = []
    resultados = []
    for i in range(execucoes):
        regr = MLPRegressor(
            hidden_layer_sizes=arq,
            max_iter=iteracoes,
            activation='tanh',
            solver='adam',
            learning_rate='adaptive',
            n_iter_no_change=iteracoes,
            verbose=False
        )
        modelo = regr.fit(x, y)
        erro_final = modelo.loss_curve_[-1]
        erros.append(erro_final)
        modelos.append(modelo)
        resultados.append(modelo.predict(x))
    return erros, modelos, resultados

for nome_arquivo in arquivos:
    print(f'\n==== Testando {nome_arquivo} ====')
    arquivo = np.load(nome_arquivo)
    x = arquivo[0]
    y = np.ravel(MaxAbsScaler().fit(arquivo[1]).transform(arquivo[1]))

    for arq in arquiteturas:
        print(f'\nArquitetura: {arq}')
        erros, modelos, resultados = executar_teste(x, y, arq)

        media = round(statistics.mean(erros), 6)
        desvio = round(statistics.stdev(erros), 6)
        print(f'Média do erro final: {media}')
        print(f'Desvio padrão: {desvio}')

        idx_melhor = np.argmin(erros)
        melhor_modelo = modelos[idx_melhor]
        y_pred = resultados[idx_melhor]

        plt.figure(figsize=[14, 7])
        plt.suptitle(f'{nome_arquivo} | Arquitetura: {arq} | Erro: {round(erros[idx_melhor], 5)}', fontsize=14)

        plt.subplot(1, 3, 1)
        plt.title('Função Original')
        plt.plot(x, y, color='green')

        plt.subplot(1, 3, 2)
        plt.title('Curva de Erro')
        plt.plot(melhor_modelo.loss_curve_, color='red')

        plt.subplot(1, 3, 3)
        plt.title('Original vs Aproximada')
        plt.plot(x, y, linewidth=1, color='green', label='Original')
        plt.plot(x, y_pred, linewidth=2, color='blue', label='Aproximada')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{nome_arquivo}_arch_{str(arq).replace(",", "-")}.png')
        plt.close()
