# -*- coding: utf-8 -*-
"""
Este script toma las reseñas en inglés y las cruza con la información 
de géneros por libro. Calcula estadísticas por género, incluyendo:
- Total de reseñas
- Número de usuarios distintos
- Usuarios más activos por género
- Cantidad de usuarios con más de 1000 reseñas en un solo género
"""

import json
from collections import defaultdict, Counter

reviews = '.json'
generos = '.json' 

# Cargar los datos de los archivos JSON
with open(reviews, 'r', encoding='utf-8') as f:
    reviews = [json.loads(line) for line in f]

with open(generos, 'r', encoding='utf-8') as f:
    genres = [json.loads(line) for line in f]

# Crear un diccionario para mapear book_id a sus géneros
book_genres = {}
for entry in genres:
    book_genres[entry['book_id']] = list(entry['genres'].keys())

# Crear estructuras para contar reviews y usuarios por género
genre_review_count = defaultdict(int)
genre_user_count = defaultdict(set)
genre_user_reviews = defaultdict(Counter)

# Procesar las reseñas
for review in reviews:
    book_id = review['book_id']
    user_id = review['user_id']
    
    if book_id in book_genres:
        for genre in book_genres[book_id]:
            genre_review_count[genre] += 1
            genre_user_count[genre].add(user_id)
            genre_user_reviews[genre][user_id] += 1

# Preparar los resultados
results = []
for genre, count in genre_review_count.items():
    top_500_users = genre_user_reviews[genre].most_common(500)
    users_with_more_than_1000_reviews = sum(1 for user, review_count in genre_user_reviews[genre].items() if review_count > 1000)
    results.append({
        'genre': genre,
        'review_count': count,
        'distinct_user_count': len(genre_user_count[genre]),
        'top_500_users': [{'user_id': user, 'review_count': review_count} for user, review_count in top_500_users],
        'users_with_more_than_1000_reviews': users_with_more_than_1000_reviews
    })

# Guardar en un archivo JSON
output_file = '.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

print(f"El archivo de resumen se ha guardado en {output_file}")
