# LDA-Goodreads-Reviews
Trabajo de Fin de Grado, uso de modelos LDA, y aplicación a reseñas de Goodreads.

Este repositorio contiene el código y documentación desarrollados como parte del Trabajo de Fin de Grado (TFG), titulado **"Aplicación de modelos LDA a la identificación de tópicos en Reseñas"**.

El objetivo principal es aplicar técnicas de modelado de tópicos, concretamente el modelo LDA (*Latent Dirichlet Allocation*), para descubrir tópicos latentes en reseñas de libros extraídas de la plataforma Goodreads. Se analiza la distribución de tópicos según géneros y valoraciones, se evalúa la convergencia del modelo y se exploran relaciones entre tópicos mediante divergencia KL y visualizaciones.

---

## 📦 Dataset

Las reseñas utilizadas provienen de una fuentes pública:

1. **Goodreads Book Reviews Dataset**  
   > Ahmad. (2023). *Goodreads Book Reviews*. Kaggle.  
   > [https://www.kaggle.com/datasets/pypiahmad/goodreads-book-reviews1](https://www.kaggle.com/datasets/pypiahmad/goodreads-book-reviews1)

   Contiene más de 15 millones de reseñas de usuarios. Para este trabajo se utilizaron las siguientes columnas:
   - `user_id`
   - `book_id`
   - `review_text`
   - `rating` (filtrando aquellas con puntuación 0)

Citas:  
   - Mengting Wan, Julian McAuley, "[Item Recommendation on Monotonic Behavior Chains](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/recsys18_mwan.pdf)", in RecSys'18. [[bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/recsys/WanM18)]
- Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "[Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://github.com/MengtingWan/mengtingwan.github.io/raw/master/paper/acl19_mwan.pdf)", in ACL'19. [[bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/acl/WanMNM19)]
 

---

## 🧠 Estructura del Proyecto y Orden de Ejecución

### 1. **Filtrado Inicial y Selección de Datos**

- `reduce_only_english_no_cero.py`:  
  Elimina reseñas que no están en inglés o con puntuación 0.

- `genero_reviews.py`:  
  Detecta el género de cada reseña.

- `Only_{}_review.py`:  
  Filtra las reseñas para trabajar con un único género específico (p.ej. solo "fantasía").

### 2. **Preprocesado de Texto**

- `Preprocesado.py`:  
  Limpieza del texto (URLs, stopwords, saltos de línea...) usando `nltk`, `gensim` y `spacy`.

- `Filtrado.R`:  
  Elimina palabras extremadamente frecuentes o raras del vocabulario.

- `filtrado_puntuaciones_y_minimo_reseñas.R`:  
  Filtra usuarios que no han usado todas las puntuaciones o que tienen muy pocas reseñas.

### 3. **Preparación para el Modelo LDA**

- `Elementos_LDA.R`:  
  Estructura los documentos y genera los elementos necesarios para la realización del modelo LDA.

### 4. **Modelado LDA**

- `LDA_Lotes.py`:  
  Aplicación del modelo LDA por lotes para grandes volúmenes de datos.

- `Unir_Lotes_LDA.py`:  
  Reagrupación de los lotes en una única salida final.

- `LDA.py`:  
  Versión alternativa del modelo LDA completo sobre el conjunto procesado.

### 5. **Análisis y Visualización de Resultados**

- `Distribución_tópicos_puntuación.py`:  
  Analiza cómo se distribuyen los tópicos en función de la puntuación dada por los usuarios.

- `Convergencia.py`:  
  Mide la convergencia de la proporción de los distintos tópicos en las reseñas según la puntuación asignada.

- `Divergencia_KL.py`:  
  Calcula la divergencia de Kullback-Leibler entre tópicos para detectar tópicos transversales, así mismo, realiza un cambio de escala para una mayor visualización.

- `Nube_Palabras.py`:  
  Generación de nubes de palabras representativas por tópico y sentimiento.

---

## 📁 Estructura de Carpetas

Todos los scripts de procesamiento, modelado y análisis se encuentran organizados en la carpeta [`scripts/`](./scripts).  
Allí podrás encontrar los archivos `.py` y `.R` utilizados en cada una de las fases descritas anteriormente.


## 🔧 Requisitos

### Python (>= 3.8)

Instala las librerías necesarias ejecutando:

```bash
pip install pandas numpy matplotlib seaborn wordcloud scipy gensim nltk spacy
```
También es necesario descargar recursos adicionales para nltk y spaCy. Ejecuta el siguiente código en tu entorno Python:
import nltk
```bash
nltk.download('stopwords')

import spacy
spacy.cli.download("en_core_web_sm")
```

### R
Instala los siguientes paquetes ejecutando este código en una consola de R:
```bash
install.packages(c("tidyverse", "tm", "progress", "rstudioapi", "stopwords", "dplyr"))
```

----------------Contacto---------------------

Para cualquier pregunta, sugerencia o problema relacionado con el uso de este repositorio, por favor contacta con [adriobre9@gmail.com].
