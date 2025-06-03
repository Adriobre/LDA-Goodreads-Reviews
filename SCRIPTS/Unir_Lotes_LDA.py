"""
Script para cargar, fusionar y guardar resultados parciales de un modelo LDA
almacenados en archivos JSON. 

"""


import json
import numpy as np

def load_json(filename):
    '''
    Carga un archivo JSON y lo convierte en un diccionario de NumPy arrays

    Parameters
    ----------
    filename : Archivo.

    Returns
    -------
    dict
    '''

    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return {key: np.array(value) for key, value in data.items()}
    except FileNotFoundError:
        print(f"Archivo {filename} no encontrado.")
        return None

def merge_results(first_part, second_part):
    '''
    Fusiona los resultados 

    Parameters
    ----------
    first_part : Archivo con los primeros lotes.
    second_part : Archivo con los últimos lotes.

    Returns
    -------
    merged_results : Archivo fusionado.
    '''

    merged_results = {}

    for key in first_part.keys():
        shape_first = first_part[key].shape
        shape_second = second_part[key].shape

        if shape_first != shape_second:
            print(f"Dimensiones incompatibles en {key}. No se pueden fusionar.")
            continue
        
        # Manejo específico según la estructura de cada matriz
        if key == "ust":
            merged_results[key] = np.where(first_part[key] != 0, first_part[key], second_part[key])  
        elif key == "stw":
            merged_results[key] = np.where(first_part[key] != 0, first_part[key], second_part[key])  
        elif key == "ta":
            merged_results[key] = np.where(first_part[key] != 0, first_part[key], second_part[key])  
        else:
            print(f"Clave desconocida: {key}, ignorando...")

    return merged_results

def save_json(data, filename):
    """Guarda los datos combinados en un archivo JSON"""
    with open(filename, "w") as f:
        json.dump({key: value.tolist() for key, value in data.items()}, f)

if __name__ == "__main__":
    # Cargar los primeros 5 lotes (0-5)
    first_part = load_json("LDA_lote_5.json")

    # Cargar los siguientes 10 lotes (6-15)
    second_part = load_json("LDA_lote_15.json")

    if first_part and second_part:
        # Fusionar los resultados respetando la estructura de cada matriz
        final_results = merge_results(first_part, second_part)

        # Guardar el resultado final
        save_json(final_results, " .json")

        print("Resultados fusionados y guardados en LDA_final.json")
    else:
        print("No se pudieron fusionar los resultados porque falta un archivo.")
