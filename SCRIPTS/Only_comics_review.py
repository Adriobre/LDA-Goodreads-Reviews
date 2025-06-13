# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:25:03 2025

@author: mfreschi
"""

import csv
import ast

# Primero, leemos b.json y extraemos los book_id que pertenecen a la categoría 'comics, graphic'
comics_book_ids = set()
with open('./data_final/goodreads_book_genres_initial.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # ignoramos líneas vacías
            # Usamos ast.literal_eval para convertir la cadena a diccionario (ya que se usan comillas simples)
            registro = ast.literal_eval(line)
            # Verificamos si en el diccionario 'genres' existe la clave 'comics, graphic'
            if 'comics, graphic' in registro.get('genres', {}):
                comics_book_ids.add(registro['book_id'])

# Ahora leemos a.json y filtramos las reviews cuyo book_id esté en el conjunto anterior
reviews_filtradas = []
with open('./data_final/reviews_english_no_cero.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            review = ast.literal_eval(line)
            if review.get('book_id') in comics_book_ids:
                reviews_filtradas.append(review)

# Escribimos las reviews filtradas en un CSV
with open('./data_final/reviews_comics.csv', 'w', encoding='utf-8', newline='') as csvfile:
    # Definimos los campos que queremos incluir (ajusta según lo que necesites)
    fieldnames = ['user_id', 'book_id', 'rating', 'review_text', 'date_added']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for review in reviews_filtradas:
        writer.writerow(review)

print("Se ha creado el fichero CSV con las reviews de libros de la categoría 'comics, graphic'.")