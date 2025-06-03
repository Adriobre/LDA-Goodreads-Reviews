"""
Preprocesamiento de reseñas para análisis de texto.

Este script limpia, tokeniza y lematiza las reseñas de libros, eliminando ruido
y preparando los datos para su posterior análisis.

"""

import pandas as pd
import re
import gensim 
import nltk 
from nltk.corpus import stopwords 
import spacy

file_path = " "
data = pd.read_csv(file_path)

# Convertir a DataFrame
data = pd.DataFrame(data)
#Cambiar el nombre de las columnas
data.columns=['user_id', 'book_id', 'rate', 'review', 'date_added']

# Eliminar reseñas duplicadas
data = data.drop_duplicates(subset='review')

# Descargar stopwords de NLTK antes de usarlas
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Cargar el modelo de spaCy antes de la lematización
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Preprocesar los datos de texto 
def preprocess_text(text): 
    '''
    Preprocesa el texto eliminando URLs, saltos de línea, tabulaciones, puntuación, caracteres especiales, etc.

    Parameters
    ----------
    text : str
        Reseña de texto que será procesada.

    Returns
    -------
    str
        Texto limpio y preprocesado.
    '''
    #Elimina URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Eliminar saltos de línea y tabulaciones
    text = re.sub(r'[\n\r\t]', ' ', text)
    #Eliminar carácteres repetidos mas de 4 veces (ej: aaaaaaaaaaaa)
    text = re.sub(r'(.)\1{4,}', ' ', text)
    # Eliminar puntuación
    text = re.sub(r'[^\w\s]', ' ', text)
    #Eliminar _
    #text = re.sub(r'_', ' ', text)
    # Eliminar caracteres no ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Eliminar espacios adicionales 
    text = re.sub(r'\s+', ' ', text) 
    # Eliminar correos electrónicos 
    text = re.sub(r'\S*@\S*\s?', '', text) 
    # Eliminar apóstrofos 
    text = re.sub(r'\'', '', text)
    # Eliminar caracteres no alfabéticos 
    text = re.sub(r'[^a-zA-Z]', ' ', text) 
    # Convertir a minúsculas 
    text = text.lower() 
    return text 

# Tokenización y eliminación de stopwords
def tokenize(text):
    '''
    Tokeniza el texto, convirtiéndolo en una lista de palabras y elimina las palabras vacías (stopwords).

    Parameters
    ----------
    text : str
        Reseña procesada que será tokenizada.

    Returns
    -------
    list
        Lista de tokens (palabras) filtradas sin stopwords.

    '''
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def lemmatize(tokens):
    '''
    Lematiza los tokens usando el modelo spaCy para obtener las formas base de las palabras.

    Parameters
    ----------
    tokens : list
        Lista de tokens (palabras) que serán lematizadas.

    Returns
    -------
    list
        Lista de lemas correspondientes a los tokens.
    '''
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]


data['cleaned_text'] = data['review'].apply(preprocess_text)

data['tokens'] = data['cleaned_text'].apply(tokenize)

data['lemmas'] = data['tokens'].apply(lemmatize)

# Unir de nuevo
data['text'] = data['lemmas'].apply(lambda x: ' '.join(x))

#Eliminar text en blanco
data= data[data['text'].str.strip() != '']

#Anadir columna id
data.insert(0, 'id', range(1, len(data)+1))

#Filtrar por las columnas requeridas
data = data[['id', 'user_id', 'book_id', 'rate', 'date_added', 'text']]

print(data)

#Guardamos la base
data.to_csv(" ", index = False)