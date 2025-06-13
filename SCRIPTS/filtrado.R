# ------------------------------------------------------------------------------
# Script para el análisis y filtrado de reseñas textuales
# Se realiza una tokenización de texto y un filtrado cálculando la frecuencias por documento y corpus
# de las palabras. Finalmente se preparan los datos para el modelo LDA.
# ------------------------------------------------------------------------------

library(tidyverse)
library(tm)
library(progress)
library(rstudioapi)
library(stopwords) 

# Obtener la ruta del archivo activo
act_path <- dirname(rstudioapi::getActiveDocumentContext()$path)

# Obtención del archivo (.csv)
# Con estructura: id, user_id, book_id, rate, date_added, text
review_nueva <- read.csv(file.choose())

# Tokenizar el texto
tokenize <- function(text){
  new_text <- str_split(text, " ")[[1]]
  return(new_text)
}


# Iniciar barra de progreso
pb <- progress::progress_bar$new(
  total = nrow(review_nueva), 
  format = "  [:bar] :percent ETA: :eta",
  clear = FALSE, 
  width = 50
)

review_nueva <- review_nueva %>% 
  mutate(token = map(.x = text, .f = function(x) {
    # Actualiza la barra de progreso
    pb$tick()  
    tokenize(x)  
  }))


# Expandir tokens y eliminar columna original de texto
review_nueva <- review_nueva %>% select(-text) %>% unnest(cols=c(token))

## Obtener stopwords en inglés
stopw <- stopwords("en", source = "stopwords-iso")
# Normalizar caracteres
stopw <- unique(chartr("áéíóúàèìòùü", "aeiouaeiouu", stopw))  

## Frecuencia de los tokens por review
freq_review <- review_nueva %>% group_by(id, token) %>% summarize(freq = n())

# Calcular la frecuencia global de los tokens en el corpus
nreview <- length(unique(review_nueva$id))
freq_corpus <- freq_review %>% select(id, token) %>% group_by(token) %>% summarise(freq=n())
freq_corpus <- freq_corpus %>% mutate(prop = freq/nreview)

# Palabras más frecuentes
freq_corpus$token[which(freq_corpus$prop > 0.2)] # Cambiar 0.2 por el umbral deseado

# Palabras menos frecuentes
freq_corpus$token[which(freq_corpus$prop < 0.005)] # Cambiar 0.005 por el umbral deseado
sum(freq_corpus$prop < 0.005)

# Lista de nombres propios presentes en las reviews
nombres <- c("adam", "alex", "anna", "anne", "ben", "charlie", "daniel", "david", 
             "elizabeth", "emma", "george", "harry", "henry", "kate", "jennifer", 
             "jake", "james", "jane", "joe", "john", "jack", "jackson", "michael", 
             "paul", "peter", "rachel", "robert", "sarah", "scott", "ryan", "simon", 
             "smith", "thomas", "tom", "turner", "william")

# Revisar palabras formadas por menos de 3 letras y con una frecuencia no muy baja
freq_corpus$token[str_length(freq_corpus$token) < 3 & freq_corpus$prop >= 0.005] # Cambiar 0.005 por el umbral deseado

# Eliminar palabras poco frecuentes y con pocas letras
freq_filter <- freq_corpus %>% filter(prop >=  0.005 )
# Filtrado adicional por longitud mínima y palabras irrelevantes (excepto "tv", cambiarlo por las palabras deseadas)
freq_filter <- freq_filter %>% filter(!(str_length(token) < 3 & prop >= 0.005 & !(token %in% c("tv")))) 

# Eliminar los nombres propios de la lista
freq_filter <- freq_filter %>% filter(!(token %in% nombres))

# Eliminar stopwords en inglés
freq_filter <- freq_filter %>% filter(!(token %in% stopw))

# Eliminar las palabras muy frecuentes, salvo excepciones significativas
freq_filter <- freq_filter %>% filter(!(prop > 0.13 & !(token %in% c("love", "life", "time","author" , "enjoy" ,"great" ,"look","little","never" ,  "new" , "review","start", "friend","help", "man"  , "people" ,"reader","world"))))

# Eliminar stopwords adicionales específicas
freq_filter <- freq_filter %>% filter(!(token %in% c("anyone", "anything", "anywhere", "everyone", "everything", "everywhere", "nothing", "someone", "across", "along", "around", "behind", "inside", "onto", "outside", "under", "although", "because", "though", "whether", "almost", "always", "already", "exactly", "finally", "maybe", "mostly", "perhaps", "probably", "simply", "soon", "therefore", "thus", "usually", "yet", "can", "could", "may", "might", "must", "either", "every", "neither", "none", "other", "whose","aka", "awhile", "discuss")))

# Reconstrucción del texto filtrado por reseña
review_nueva2 <- review_nueva %>% filter(token %in% freq_filter$token)
review_nueva2 <- review_nueva2 %>% group_by(id, user_id,book_id, rate,date_added) %>% 
  summarise(text=paste0(token, collapse = " "))

# Reordenar columnas
review_nueva2 <- review_nueva2[,c("id", 'user_id', 'book_id', 'rate', 'date_added', 'text')]

# Crear matríz de terminos para un análisis posterior
corpus <- Corpus(VectorSource(review_nueva2$text))
tdm <- TermDocumentMatrix(corpus, control=list(wordLengths=c(1,Inf)))

# Guardar el resultado final
write.csv2(review_nueva2, " .csv")

# Guardar estadísticas
write.csv(freq_corpus, " .csv")
write.csv(freq_review, " .csv")
write.csv(freq_filter, " .csv")
