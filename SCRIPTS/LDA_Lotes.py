"""
Implementación de un modelo de caracterización de tópicos LDA por lotes,
 usando Gibbs Sampling.
"""

import numpy as np
import pandas as pd
import json
import datetime
from scipy.special import loggamma
import math

class SentimentSTLDA:
    def __init__(self, nT, rate, user, matrix_words, seed=123):
        '''
        Inicializa el modelo LDA

        Parameters
        ----------
        nT : Número de tópicos.
        rate : Puntuaciones de cada reseña.
        user : Usuarios.
        matrix_words : Matriz de palabras.
        '''
        self.nT = nT
        self.rate = rate
        self.users = user
        self.dw = matrix_words
        self.seed = seed

        # Validar que los datos se pasen correctamente
        if rate is None or user is None or matrix_words is None:
            raise ValueError("Se requieren rate, user y matrix_words para inicializar el modelo.")

        # Mapear sentimientos directamente sin categorizarlos
        unique_sentiments = np.unique(rate)
        sentiment_mapping = {val: idx for idx, val in enumerate(unique_sentiments)}
        self.sa_true = np.array([sentiment_mapping[val] for val in rate])  # Mapear rate a índices consecutivos
        self.nS = len(unique_sentiments)  # Número de sentimientos

        # Número total de usuarios y palabras
        self.nU = len(np.unique(user))
        self.nW = self.dw.shape[1]
        self.nD = self.dw.shape[0]

    def init_gibbs_sampler(self):
        '''
        Inicialización de ta (asignación de tópicos) de forma aleatoria y 
        devuelve los conteos iniciales de ust y stw
        -------
        ta :  Asignación inicial de tópicos a documentos.
        ust : Conteo de usuarios por sentimiento y tópico.
        stw : Conteo de palabras por sentimiento y tópico.
        '''
        np.random.seed(self.seed)
        ta = np.random.choice(np.arange(self.nT), replace=True, size=self.nD)

        ust = np.zeros([self.nU, self.nS, self.nT])
        for u in range(self.nU):
            for s in range(self.nS):
                for t in range(self.nT):
                    ust[u, s, t] = np.sum((self.users == u) & (self.sa_true == s) & (ta == t))

        stw = np.zeros([self.nS, self.nT, self.nW])
        for s in range(self.nS):
            for t in range(self.nT):
                stw[s, t, :] = np.sum(self.dw[(self.sa_true == s) & (ta == t)], axis=0)

        return ta, ust, stw
    
    def recompute_counts(self, ta):
        '''
        Cálculo los conteos de ust y stw a partir de la asignación de tópicos actual.
        
        Parameters:
        ----------
        ta : Asignación actualizada de tópicos a documentos.
        '''
        
        ust = np.zeros([self.nU, self.nS, self.nT])
        for u in range(self.nU):
            for s in range(self.nS):
                for t in range(self.nT):
                    ust[u, s, t] = np.sum((self.users == u) & (self.sa_true == s) & (ta == t))

        stw = np.zeros([self.nS, self.nT, self.nW])
        for s in range(self.nS):
            for t in range(self.nT):
                stw[s, t, :] = np.sum(self.dw[(self.sa_true == s) & (ta == t)], axis=0)

        return ust, stw

    def gibbs_iteration(self, ta, ust, stw, alpha, eta):
        '''
        Ejecución de una iteración de Gibbs

        Parameters
        ----------
        ta : Asignación actual de tópicos.
        ust : Matriz de conteo usuario-sentimiento-tópico.
        stw : Matriz de conteo sentimiento-tópico-palabra.
        alpha : Hiperparámetro para la distribución Dirichlet.
        eta : Hiperparámetro para la distribución Dirichlet.
        '''
        for d in range(self.nD):
            # Obtener estado actual del documento d
            u = self.users[d]
            s = self.sa_true[d]
            t = ta[d]

            # Eliminar los conteos
            ust[u, s, t] -= 1
            stw[s, t] -= self.dw[d,]

            # Calcular probabilidades
            us_to_t = np.log((alpha + ust[u, s]) / (self.nT * alpha + np.sum(ust[u, s])))
            w_to_t = loggamma(np.sum(stw[s,] + eta, axis=1))
            w_to_t -= np.sum(loggamma(stw[s,] + eta), axis=1)
            w_to_t += np.sum(loggamma(stw[s,] + eta + self.dw[d,]), axis=1)
            w_to_t -= loggamma(np.sum(stw[s,] + eta, axis=1) + np.sum(self.dw[d]))

            probs = np.exp((us_to_t + w_to_t) - np.max(us_to_t + w_to_t))
            probs /= np.sum(probs)

            t_new = np.random.choice(np.arange(self.nT), p=probs)

            # Agregar de nuevo a los conteos
            ta[d] = t_new
            ust[u, s, t_new] += 1
            stw[s, t_new] += self.dw[d,]

        return ta, ust, stw

    def run_gibbs_sampler(self, niter, nburn, nthin, nlot, alpha, eta,nupdate,
                          use_previous = False, previous_inference = None, lastlot = None):
        '''
        Ejecución del muestreo de Gibb por lotes

        Parameters
        ----------
        niter : Número total de iteraciones.
        nburn : Número de iteraciones de calentamiento (burn-in).
        nthin : Intervalo de muestreo.
        nlot : Tamaño de cada lote de iteraciones.
        alpha : Parámetro Dirichlet para tópicos.
        eta : Parámetro Dirichlet para palabras.
        '''
        if use_previous and previous_inference is not None:
            if lastlot is None:
                lastlot = 2 #Se pondría el último lote ejecutado 
            index = int(nlot // nthin * lastlot)
            print("Inicializando a partir de resultados previos, usando columna", index)
            ta = previous_inference["ta"][:, index - 1].copy()
            ust, stw = self.recompute_counts(ta)
        else:
            print("Inicializando aleatoriamente")
            ta, ust, stw = self.init_gibbs_sampler()

        nsaved = np.floor(niter / nthin).astype(int) - 1
        results_ust = np.zeros([self.nU, self.nS, self.nT, nsaved])
        results_stw = np.zeros([self.nS, self.nT, self.nW, nsaved])
        results_ta = np.zeros([self.nD, nsaved])

        iter_start = datetime.datetime.now()
        
        last_complete_lot = -1
        total_lots = (niter // nlot) + 1
        
        #En caso de un error, for lot in range(lastlot + 1, total_lots):
        for lot in range(total_lots):
            try: 
              np.random.seed(self.seed + lot)
              if lot < 1:
                  niterstar = 0
                  niternd = nburn 
              else:
                  niterstar = nburn + (lot - 1) * nlot + 1 
                  niternd = nburn + lot * nlot 
            
              for iter in range(niterstar , niternd + 1):
                  ta, ust, stw = self.gibbs_iteration(ta, ust, stw, alpha, eta)

                  if (lot > 0) & (iter > nburn) & (iter % nthin == 0):
                      idx = ((iter - nburn) // nthin) - 1
                      for s in range(self.nS):
                          results_ust[:, s, :, idx] = np.apply_along_axis(lambda x: x / np.sum(x), arr=ust[:, s, :], axis=1)
                          results_stw[s, :, :, idx] = np.apply_along_axis(lambda x: x / np.sum(x), arr=stw[s], axis=1)

                      results_ta[:, idx] = ta

                  if (iter % nupdate) == 0:
                      print(f'Finalizada iter {iter} a las {datetime.datetime.now()}')
                      print(f'Tiempo por iter: {datetime.datetime.now() - iter_start}')
                      iter_start = datetime.datetime.now()
                      
              self.results_ust = results_ust  # Distribución de sentimientos por usuario y tópico
              self.results_stw = results_stw  # Distribución de palabras por sentimiento y tópico
              self.results_ta = results_ta    # Asignaciones de tópicos a documentos a lo largo de las iteraciones.
              self.save_partial_results(lot)  # Llamar a la función para guardar el lote
            
              last_complete_lot = lot
              #inference_result = { "ust": results_ust, "stw": results_stw,"ta": results_ta}
              print( f"Finalizado lot {lot} a las {datetime.datetime.now()}")
            
            except Exception as e:
              print(f"Error en el lote {lot} : {e}")
              lot = last_complete_lot + 1
              np.random.seed(self.seed + lot)
              print(f"Retomar el lote {lot}" )
    def save_partial_results(self, lot):
        '''
        Guardado de los resultados en un archivo JSON después de cada lote.

        Parameters
        ----------
        lot : Número de lote completado..
        '''

        results_data = {
               "ust": self.results_ust.tolist(),
               "stw": self.results_stw.tolist(),
               "ta": self.results_ta.tolist(),
               "last_complete_lot": lot
               }

        with open(f"LDA_lote_{lot}.json", "w") as f:
              json.dump(results_data, f)

        print(f"Resultados del lote {lot} guardados exitosamente.")
            


if __name__ == "__main__":
    matrix_words = pd.read_csv(" .csv").values
    rate = pd.read_csv(" .csv")["review_nueva.rate"].values
    user = pd.read_csv(" .csv")["user2"].values

    #Iniciar de los parámetros
    nT, niter, nburn, nthin, nlot = 10, 150, 10, 1, 50
    alpha, eta, seed = 1, 1, 123
    nupdate = max(1, math.floor(niter / 10))

    model = SentimentSTLDA(nT, rate, user, matrix_words, seed)
    
    #Si se tiene un resultado previo:
    #previous_inference = {}
    #model.run_gibbs_sampler(niter, nburn, nthin, nlot, alpha, eta, nupdate,
    #                          use_previous = True, previous_inference = previous_inference, lastlot = [valor])
    
    model.run_gibbs_sampler(niter, nburn, nthin, nlot, alpha, eta, nupdate)
    
    # Guardar resultados
    results_ust = model.results_ust
    results_stw = model.results_stw
    results_ta = model.results_ta

    with open("LDA.json", "w") as f:
         json.dump({
              "ust": results_ust.tolist(),
              "stw": results_stw.tolist(),
              "ta": results_ta.tolist()
         }, f)

    print("Resultados guardados exitosamente en formato json.")
