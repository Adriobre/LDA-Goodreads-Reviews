"""
Este script implementa un modelo LDA que incorpora sentimientos por documento y
usuarios. Usa Gibbs Sampling colapsado para la inferencia de tópicos y distribuciones.

Los resultados se guardan en un archivo JSON.
"""

import datetime
import numpy as np
import pandas as pd
from scipy.special import loggamma
import math



class sentiment_stLDA:
    """
    Modelo de LDA.
    Args:
        nT (int): número de tópicos
        rate (array): calificaciones (sentimientos) por documento
        user (array): usuarios por documento
        matrix_words (2D array): matriz documento-palabra
    """

    def __init__(self, nT=None, rate=None, user=None, matrix_words=None, seed=None):
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
        self.nS = len(unique_sentiments)  # Actualizar el número de sentimientos


        # Número total de usuarios y palabras
        self.nU = len(np.unique(user))
        self.nW = self.dw.shape[1]
        self.nD = self.dw.shape[0] 
        #self.nS = len(np.unique(rate))
        
        
            
    #Inicializar parametros        
    def init_collapse_gibbs_sampler(self):

        np.random.seed(self.seed)
        ta = np.random.choice(np.arange(self.nT), replace = True, size = self.nD)

        ust = np.zeros([self.nU, self.nS, self.nT])
        for u in range(self.nU):
            for s in range(self.nS):
                for t in range(self.nT):
                    ust[u, s, t] = np.sum( (self.users == u) & (self.sa_true == s) & (ta == t) )

        stw = np.zeros([self.nS, self.nT, self.nW])
        for s in range(self.nS):
            for t in range(self.nT):
                stw[s,t,:] = np.sum( self.dw[(self.sa_true == s) & 
                                            (ta == t)], axis=0 )

        return ta, ust, stw


    def iter_collapse_gibbs_sampler(self, ta, ust, stw, alpha=0.01, eta=0.01):

        for d in range(self.nD):
                # Obtención del estado actual de cocumento d
                u = self.users[d]
                s = self.sa_true[d]
                t = ta[d]

                # se elimina del conteo
                ust[u,s,t] -= 1
                stw[s,t]   -= self.dw[d,]

                # se calculan las probabilidades
                us_to_t = np.log( (alpha + ust[u, s]) / (self.nT * alpha + np.sum(ust[u, s]) ) )
                w_to_t  = loggamma( np.sum( stw[s,] + eta, axis = 1) ) 
                w_to_t -= np.sum( loggamma( stw[s,] + eta ), axis=1)
                w_to_t += np.sum( loggamma(stw[s,] + eta + self.dw[d,]), axis=1 )
                w_to_t -= loggamma( np.sum( stw[s,] + eta, axis=1 ) + np.sum(self.dw[d]) )
                probs = np.exp((us_to_t + w_to_t) - np.max(us_to_t + w_to_t))
                probs /= np.sum(probs)

                t_new = np.random.choice(np.arange(self.nT), p = probs) 

                # se añaden los conteos
                ta[d] = t_new
                ust[u,s,t_new] += 1
                stw[s,t_new,] += self.dw[d,]

        return ta, ust, stw


    def collapse_gibbs_sampler(self, niter=None, nburnin=None, nthin=None, 
        nupdate=None, alpha= None, eta=None):

        ta, ust, stw = self.init_collapse_gibbs_sampler()
                
        nsaved = np.floor(niter/nthin).astype(int) - 1
        results_ust = np.zeros([self.nU, self.nS, self.nT, nsaved]) 
        results_stw = np.zeros([self.nS, self.nT, self.nW, nsaved]) 
        results_ta  = np.zeros([self.nD, nsaved]) 

        iter_start = datetime.datetime.now()
        for i in range(niter+nburnin):
            
            ta, ust, stw = self.iter_collapse_gibbs_sampler(ta, ust, stw, alpha, eta)
                
            if (i > nburnin) & (i % nthin == 0) :
                idx = ((i - nburnin) // nthin) - 1
                for s in range(self.nS):
                    results_ust[:,s,:,idx] = np.apply_along_axis(lambda x: 
                                                                x/np.sum(x), arr=ust[:,s,:], axis=1)
                    
                    results_stw[s,:,:,idx] = np.apply_along_axis(lambda x: 
                                                                x/np.sum(x), arr=stw[s], axis=1)

                results_ta[:,idx] = ta
                    
            if (i % nupdate) == 0:
                print(f'Acabada iter {i} en tiempo {datetime.datetime.now()}')
                print(f'Tiempo por iter: {datetime.datetime.now() - iter_start}')
                iter_start = datetime.datetime.now()

        self.results_ust = results_ust #Distribución de sentimientos por usuario y tópico
        self.results_stw = results_stw #Distribución de palabras por sentimiento y tópico
        self.results_ta  = results_ta  #Asignaciones de tópicos a documentos a lo largo de las iteraciones.




############################################################################################
############################################################################################
# Guardar resultados
if __name__ == "__main__":
    # Cargar los archivos CSV
    matrix_words = pd.read_csv(" .csv").values
    rate = pd.read_csv(" .csv")['x'].values
    user = pd.read_csv(".csv")['user2'].values

    # Definir los parámetros (Cambiar los parámetros a los deseados)
    nT = 4
    niter = 3000
    alpha = 0.001
    eta = 0.01
    seed = 123
    nupdate = max(1, math.floor(niter / 10))
    nthin = 5
    nburnin = 1500

    # Inicializar y ejecutar el modelo
    model = sentiment_stLDA(nT=nT, rate=rate, user=user, matrix_words=matrix_words, seed=seed)
    model.collapse_gibbs_sampler(niter=niter, nburnin=nburnin, nthin=nthin, 
        nupdate=nupdate, alpha= alpha, eta=eta)

    # Guardar resultados
    results_ust = model.results_ust
    results_stw = model.results_stw
    results_ta = model.results_ta

    # Guardar resultados como archivos
    import json

    with open(" .json", "w") as f:
         json.dump({
              "ust": results_ust.tolist(),
              "stw": results_stw.tolist(),
              "ta": results_ta.tolist()
         }, f)

    print("Resultados guardados exitosamente en formato CSV.")

