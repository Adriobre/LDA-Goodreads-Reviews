"""
Script para analizar y visualizar la distribución de asignaciones de tópicos
por sentimiento a partir de resultados de un modelo LDA de sentimiento y tópico.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from scipy.stats import mode
import matplotlib.gridspec as gridspec

# Estilo general
sns.set(style="whitegrid")

# Cargar resultados
with open(" .json", "r") as f:
    results = json.load(f)

results_ust = np.array(results["ust"])
results_stw = np.array(results["stw"])
results_ta = np.array(results["ta"])

# Cargar palabras y sentimientos
matrix_words_df = pd.read_csv(" .csv")
lista_palabras = matrix_words_df.columns.tolist()
nS, nT, nW, _ = results_stw.shape 

rate = pd.read_csv(" .csv")['x'].values

# Función para analizar y graficar
def analizar_ta_en_grid(mode_ta, sa_true):
    unique_sentiments = np.unique(sa_true)
    n_sentimientos = len(unique_sentiments)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    posiciones = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]  # Máx 5 sentimientos

    for idx, s in enumerate(unique_sentiments):
        ta_s = mode_ta[sa_true == s]
        ta_s = ta_s[~np.isnan(ta_s)].astype(int)
        counts = np.bincount(ta_s, minlength=nT)
        proportions = counts / counts.sum()

        row, col = posiciones[idx]
        ax = fig.add_subplot(gs[row, col])
        palette = sns.color_palette("husl", nT)
        sns.barplot(x=np.arange(1, nT + 1), y=proportions, palette=palette, ax=ax)

        ax.set_title(f"Sentimiento {s }", fontsize=16)
        ax.set_xlabel("Tópico", fontsize=14)
        ax.set_ylabel("Proporción", fontsize=14)
        ax.set_ylim(0, 0.5)
        ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    # plt.savefig("distribucion_topicos_sentimientos.png", dpi=300)
    plt.show()

# Ejecutar análisis
def ejecutar_analisis(results_ta, rate):
    mode_ta = mode(results_ta, axis=-1).mode
    analizar_ta_en_grid(mode_ta, rate)

ejecutar_analisis(results_ta, rate)
