"""
Script para analizar la evolución de la proporción de documentos asignados
a cada tópico a lo largo de las iteraciones del modelo LDA, separando por sentimiento.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

# Cargar resultados desde JSON
with open(" .json", "r") as f:
    results = json.load(f)

# Convertir datos en arrays de NumPy
results_stw = np.array(results['stw'])  
results_ta = np.array(results['ta'])   
nS, nT, nW, nIter = results_stw.shape  

# Cargar sentimientos
rate = pd.read_csv(" .csv")['x'].values
unique_sentiments = np.unique(rate)
sentiment_mapping = {val: idx for idx, val in enumerate(unique_sentiments)}
results_true = np.array([sentiment_mapping[val] for val in rate])  

# Se calcula la moda
def calcular_moda_ta(results_ta):
    nD, nIter = results_ta.shape
    moda_ta = np.zeros((nD, nIter), dtype=int)
    for d in range(nD):
        conteo = np.zeros(nT, dtype=int)  
        for i in range(nIter):
            aux = int(results_ta[d, i])  
            conteo[aux] += 1   
            moda_ta[d, i] = np.argmax(conteo)  
    return moda_ta

#Se calcula la proporción
def calcular_proporcion_por_sentimiento(moda_ta):
    proporcion_st = np.zeros((nS, nT, nIter))
    for s in range(nS):
        docs_sentimiento = np.where(results_true == s)[0]  
        for i in range(nIter):
            topicos_asignados = moda_ta[docs_sentimiento, i]
            counts = np.bincount(topicos_asignados, minlength=nT)  
            proporcion_st[s, :, i] = counts / len(docs_sentimiento)  
    return proporcion_st

def graficar_proporcion_por_sentimiento(proporcion_st):
    '''
    Distribuye los subplots como: 2 arriba, 1 en medio, 2 abajo.
    '''
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

    posiciones = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]
    for s in range(nS):
        row, col = posiciones[s]
        ax = plt.subplot(gs[row, col])
        for t in range(nT):
            ax.plot(proporcion_st[s, t, :], label=f'Tópico {t + 1}')  # Tópicos desde 1
        ax.set_title(f'Sentimiento {s + 1}', fontsize=16)
        ax.set_xlabel("Iteraciones", fontsize=14)
        ax.set_ylabel("Proporción de documentos", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=10)

    plt.tight_layout()
    # plt.savefig("grafico_2_1_2_sentimientos.png", dpi=300)
    plt.show()

# Ejecutar cálculos
moda_ta = calcular_moda_ta(results_ta)
proporcion_st = calcular_proporcion_por_sentimiento(moda_ta)
graficar_proporcion_por_sentimiento(proporcion_st)
