"""
Generación de nubes de palabras para cada combinación de sentimiento y tópico
basadas en las distribuciones de palabras obtenidas de un modelo LDA entrenado 
con reseñas.
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import numpy as np

# Cargar los resultados
with open(" .json", "r") as f:
    results = json.load(f)

results_stw = np.array(results['stw'])  # (nS, nT, nW, nIter)
nS, nT, nW, _ = results_stw.shape

# Palabras del vocabulario
matrix_words_df = pd.read_csv(" .csv")
words = matrix_words_df.columns.tolist()

# Media sobre las iteraciones
mean_stw_last = np.mean(results_stw, axis=-1)

# Directorio de salida
output_dir = "Nubes_de_palabras"
os.makedirs(output_dir, exist_ok=True)

# Generar las nubes
for s in range(nS):
    for t in range(nT):
        word_frequencies = mean_stw_last[s, t, :]
        sorted_indices = np.argsort(word_frequencies)[::-1]

        freq_dict = {words[i]: word_frequencies[i] for i in sorted_indices[:1000] if word_frequencies[i] > 0}

        if not freq_dict:
            continue  # Evita errores si no hay palabras

        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=500).generate_from_frequencies(freq_dict)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        #plt.title(f'Sentimiento {s} - Tópico {t}')
        plt.tight_layout()

        filename = os.path.join(output_dir, f"Nube_sentimiento{s}_topico{t}.png")
        plt.savefig(filename)
        plt.close()

        print(f" Nube generada: Sentimiento {s}, Tópico {t}")
