# -*- coding: utf-8 -*-
"""
Este script identifica todos los libros clasificados como "comics, graphic" 
a partir de un archivo de géneros, filtra las reseñas 
y las guarda en formato CSV.
"""

import csv
import ast

# Leer b.json y extraer los book_id que pertenecen a la categoría 'comics, graphic'
comics_book_ids = set()
with open('.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  
            
            registro = ast.literal_eval(line)
            
            # Verificar si en el diccionario 'genres' existe la clave 'comics, graphic'
            if 'comics, graphic' in registro.get('genres', {}):
                comics_book_ids.add(registro['book_id'])

# Leer a.json y filtrar las reviews cuyo book_id esté en el conjunto anterior
reviews_filtradas = []
with open('.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            review = ast.literal_eval(line)
            if review.get('book_id') in comics_book_ids:
                reviews_filtradas.append(review)

# Guardar las reviews filtradas en un CSV
with open('.csv', 'w', encoding='utf-8', newline='') as csvfile:
    
    # Definir los campos a incluir (ajusta según se necesite)
    fieldnames = ['user_id', 'book_id', 'rating', 'review_text', 'date_added']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for review in reviews_filtradas:
        writer.writerow(review)

print("Se ha creado el fichero CSV con las reviews de libros de la categoría 'comics, graphic'.")
