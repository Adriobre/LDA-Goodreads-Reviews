"""
Este script calcula y analiza la divergencia KL (Kullback-Leibler) entre las 
distribuciones de palabras para cada combinación de sentimiento y tópico 
obtenida de un modelo LDA entrenado con reseñas de libros.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from itertools import product

# Cargar los resultados
with open(" .json", "r") as f:
    results = json.load(f)

# Extraer la matriz stw
results_stw = np.array(results['stw'])  # Convertir a array de NumPy 
nS, nT, nW, _ = results_stw.shape  

# Se extraen las palabras
matrix_words_df = pd.read_csv(" .csv")
words = matrix_words_df.columns.tolist()


# Corregir ceros en las distribuciones para que no de error
epsilon = 1e-10
results_stw[results_stw == 0] = epsilon  

# Normalizar distribuciones
def normalize_distributions(distributions):
    return distributions / np.sum(distributions, axis=2, keepdims=True)

results_stw = normalize_distributions(results_stw)

# Verificar si las distribuciones están normalizadas
def check_normalization(distributions):

    sums = np.sum(distributions, axis=2)  
    if np.allclose(sums, 1, atol=1e-6): 
        print(" Todas las distribuciones están correctamente normalizadas.")
    else:
        print("Hay distribuciones que NO están normalizadas.")
        print("Ejemplo de sumas por sentimiento y tópico:\n", sums[:3, :3]) 
        print("Máxima desviación detectada:", np.max(np.abs(sums - 1))) 

print("Verificando normalización de las distribuciones...")
check_normalization(results_stw)

# Función para calcular la divergencia KL
def kl_divergence(p, q):
    return np.sum(rel_entr(p, q))

# Calcular divergencia KL entre todas las combinaciones de (sentimiento, tópico)
def compute_kl_for_all_combinations(distributions):
    nS, nT, nW = distributions.shape
    st_combinations = list(product(range(nS), range(nT)))
    n_combinations = len(st_combinations)
    
    # Se crea una matriz para almacenar divergencias KL
    kl_matrix = np.zeros((n_combinations, n_combinations))
    
    # Se calcula KL para cada par de combinaciones
    for i, (s1, t1) in enumerate(st_combinations):
        for j, (s2, t2) in enumerate(st_combinations):
            if i != j:  # Evitar comparación consigo mismo
                p = distributions[s1, t1]
                q = distributions[s2, t2]
                kl_matrix[i, j] = kl_divergence(p, q)
    
    return kl_matrix, st_combinations

# Identificar combinaciones transversales
def find_transversal_combinations(kl_matrix, st_combinations, aux_percentile=25):
    umbral = np.percentile(kl_matrix[kl_matrix > 0], aux_percentile) 
    transversal_pairs = np.argwhere(kl_matrix < umbral)
    transversal_pairs = [(i, j) for i, j in transversal_pairs if i != j] 
    return transversal_pairs, umbral


if __name__ == "__main__":
    distributions_last = results_stw.mean(axis=-1)  

    # Calcular divergencia KL entre todas las combinaciones de (sentimiento, tópico)
    print("Calculando divergencia KL entre combinaciones de (sentimiento, tópico)...")
    kl_matrix, st_combinations = compute_kl_for_all_combinations(distributions_last)

    # Identificar combinaciones transversales
    print("Identificando combinaciones transversales...")
    transversal_pairs, umbral = find_transversal_combinations(kl_matrix, st_combinations)
    
    # Mostrar resultados
    print(f"Combinaciones transversales identificadas (índices): {transversal_pairs}")
    print(f"Umbral de baja divergencia KL: {umbral}")
    print("Detalles de combinaciones transversales:")
    for i, j in transversal_pairs:
        s1, t1 = st_combinations[i]
        s2, t2 = st_combinations[j]
        print(f"(Sentimiento {s1}, Tópico {t1}) <-> (Sentimiento {s2}, Tópico {t2}): KL={kl_matrix[i, j]:.4f}")

 # Graficar matriz de divergencia KL con escala logarítmica
    plt.figure(figsize=(14, 12))

    # Aplicar log(1 + x) para mejor visualización
    kl_log = np.log1p(kl_matrix)
    
    # Recoger divergencias no diagonales en lista
    kl_entries = []

    for i, (s1, t1) in enumerate(st_combinations):
        for j, (s2, t2) in enumerate(st_combinations):
            if i != j:
                kl_entries.append((kl_log[i, j], f"S{s1+1}-T{t1+1}", f"S{s2+1}-T{t2+1}"))

    # Ordenar por divergencia
    kl_entries.sort()

    # Mostrar las 50 más similares
    print("\nTop 50 combinaciones más similares (log(1 + KL)):")
    for val, label1, label2 in kl_entries[:50]:
        print(f"{label1} <-> {label2}: log(1+KL) = {val:.4f}")


    # Crear etiquetas para los ejes como "S0-T1"
    labels = [f"S{s + 1}-T{t + 1}" for (s, t) in st_combinations]

    im = plt.imshow(kl_log, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, label="log(1 + KL Divergencia)")
    plt.title("Matriz logarítmica de divergencia KL entre combinaciones (Sentimiento-Tópico)")

    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xlabel("Combinaciones (Sentimiento-Tópico)")
    plt.ylabel("Combinaciones (Sentimiento-Tópico)")

    plt.tight_layout()
    plt.show()
