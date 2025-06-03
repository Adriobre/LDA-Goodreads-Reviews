# --------------------------------------------------------
# Filtrado de la base por usuarios según sus puntuaciones.
# --------------------------------------------------------

# Cargar paquete necesario
library(dplyr)


# Obtención del dataframe
review_nueva <- read.csv(file.choose())

data <-  review_nueva


# Definir los valores posibles para rates
rates <- 1:5

# Detectar los usuarios que no han usado un rate en específico
result <- data %>%
  group_by(user_id) %>%
  summarize(missing_rates = list(setdiff(rates, unique(rate))))

# Filtrar usuarios que tienen algún rate ausente
usuarios_con_rates_faltantes <- result %>%
  filter(lengths(missing_rates) > 0)

# Mostrar el resultado
print(usuarios_con_rates_faltantes)

# Mostrar los usuarios y qué ratings les faltan (en formato legible)
usuarios_con_rates_faltantes %>%
  rowwise() %>%
  mutate(missing_rates_str = paste(unlist(missing_rates), collapse = ", ")) %>%
  select(user_id, missing_rates_str) %>%
  print()


# ------------------------------
# Filtrar usuarios incompletos
# ------------------------------

# Identificar usuarios con rates incompletos
usuarios_con_rates_faltantes <- result %>%
  filter(lengths(missing_rates) > 0)

# Identificar los user_id de esos usuarios
user_ids_a_eliminar <- usuarios_con_rates_faltantes$user_id

# Filtrar el dataframe original para excluir esos usuarios
data_filtrada <- data %>%
  filter(!user_id %in% user_ids_a_eliminar)

num_users <- length(unique(data_filtrada$user_id))
print(num_users)

# Mostrar el dataframe filtrado
print(data_filtrada)

num_reseñas_filtradas <- nrow(data_filtrada)
print(num_reseñas_filtradas)


# Guardar el resultado si es necesario
write.csv(data_filtrada, " .csv", row.names = FALSE)

# -------------------------------------------------------------------
# OPCIONAL: Filtrar por cantidad de reseñas escritas por cada usuario
# -------------------------------------------------------------------

# Filtrar usuarios que han escrito 900 o más reseñas (Cambiar el número mínimo de reseñas al que se desee)
usuarios_minimo_reseñas <- data_filtrada %>%
  group_by(user_id) %>%
  filter(n() >= 900) %>%
  ungroup()

# Contar el número de usuarios restantes
num_users <- length(unique(usuarios_minimo_reseñas$user_id))
print(num_users)

# Contar el número total de reseñas de los usuarios con ese mínimo de reseñas
total_reviews <- nrow(usuarios_minimo_reseñas)
print(total_reviews)

# Guardar el resultado
write.csv(usuarios_minimo_reseñas, " .csv", row.names = FALSE)
