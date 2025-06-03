# --------------------------------------------------------------
# Preparación de los datos necesarios para el modelo LDA (Latent Dirichlet Allocation)
# --------------------------------------------------------------

# Cargar librerías necesarias
library(tidyverse)
library(tm)
library(progress)
library(rstudioapi)

# Obtener la ruta del script actual para guardar archivos en la misma carpeta
act_path <- dirname(rstudioapi::getActiveDocumentContext()$path)

# Leer el archivo con las reseñas ya procesadas 
review_nueva <- read.csv(file.choose())

# ------------------------------------------------------
# Creación de la matriz de términos
# ------------------------------------------------------

# Crear un corpus con los textos
corpus <- Corpus(VectorSource(review_nueva$text))

# Crear una Term-Document Matrix (TDM), 
# donde: Cada fila es una palabra y cada columna es una reseña
tdm <- TermDocumentMatrix(corpus, control = list(wordLengths = c(1, Inf)))
print(tdm)  # Mostrar la matriz resultante

# ------------------------------------------------------
# Guardado de la información necesaria
# ------------------------------------------------------

# Guardar la TDM transpuesta como CSV para usarla en el modelo LDA
write.csv(
  as.data.frame(t(as.matrix(tdm))),
  " .csv",
  row.names = FALSE
)

# Guardar los rates por reseña
saveRDS(
  as.vector(review_nueva$rate),
  " .rds"
)

write.csv(
  as.vector(review_nueva$rate),
  " .csv",
  row.names = FALSE
)

# Guardar los user_id por reseña
saveRDS(
  as.vector(review_nueva$user_id),
  " .rds"
)

write.csv(
  as.vector(review_nueva$user_id),
  " .csv",
  row.names = FALSE
)

# ------------------------------------------------------
# Mapeo de user_id a formato númerico y secuencial
# (Requerido para el modelo LDA)
# ------------------------------------------------------

# Cargar user_id desde archivo RDS
user_aux <- readRDS(" .rds")

# Convertir a enteros consecutivos desde 0
user2 <- as.integer(as.factor(user_aux)) - 1

# Guardar la versión codificada
write.csv(
  data.frame(user2),
  " .csv",
  row.names = FALSE
)

# Guardar los valores únicos de usuario 
unique_values <- unique(c(user_aux, user2))
saveRDS(
  unique_values,
  " .rds"
)
