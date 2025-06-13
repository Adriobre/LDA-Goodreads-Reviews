# -*- coding: utf-8 -*-
"""
Este scrpt filtra reseñas en inglés con rating no nulo y guarda los resultados en un nuevo archivo JSON.
"""

import json
from langdetect import detect

#Longitud mínima de una reseña
min_text_length = 2 

def is_in_english(quote):
  """
    Verifica si una reseña está escrita en inglés.
    Utiliza la librería langdetect para detectar el idioma.
    """
  lang= 'it'
  try:
      lang = detect(quote)
  except:
      print('Lengua no reconocida')
  return False if ((lang!= 'en') or len(quote.split()) < min_text_length) else True

#Obtener la base originar y crear base final
with open('.json', 'r') as sourcefile, open('.json', 'w') as destfile:
    intIndex=0
    for s in sourcefile:
        data = json.loads(s)

      # Eliminar información irrelevantes para el análisis
        data.pop('review_id')
        data.pop('date_updated')
        data.pop('read_at')
        data.pop('started_at')
        data.pop('n_votes')
        data.pop('n_comments')
      
        #Verificar el idioma
        text_to_know=data['review_text']
      
      # Verificar si la reseña está en inglés
        boolean_answer = is_in_english(text_to_know)
        if(boolean_answer):
          
          #Verificar que el rating no sea 0
            if data['rating']!=0 :
                json.dump(data, destfile)
                destfile.write('\n')
                intIndex+=1
              
    print("New database contains: ", str(intIndex), " entries")
