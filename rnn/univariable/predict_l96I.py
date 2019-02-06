from . import lstm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from time import gmtime, strftime
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import netCDF4 as nc4

class PredictL96I():

    def __init__(self, name, epochs, seq_len, dropout, lrate, activar, optimizar, perdida):
        #print(data)
        self.name = name
        self.epochs = epochs
        self.seq_len = seq_len
        self.drpt = dropout
        self.lrate = lrate
        self.activacion = activar
        self.optimizacion = optimizar
        self.perdida = perdida

        print(self.activacion + " " + self.optimizacion + " " + self.perdida)
        #print(self.optimizacion)
        #print(self.perdida)

        #self.X_train, self.y_train, self.X_test, self.y_test = lstm.load_data('mycsv.csv', self.seq_len, False)       
        self.X_train, self.y_train, self.X_test, self.y_test = lstm.load_data(self.name, self.seq_len, False)
        # Soluciona el problema de hilos feed dict lstm_:0 etc
        import keras.backend.tensorflow_backend
        #config = tf.ConfigProto()
        #config.intra_op_parallelism_threads = 0
        #config.inter_op_parallelism_threads = 0
        if keras.backend.tensorflow_backend._SESSION:
           #import tensorflow as tf
           tf.reset_default_graph()
           keras.backend.tensorflow_backend._SESSION.close()
           keras.backend.tensorflow_backend._SESSION = None

    def start_prediction(self):
        #model = lstm.build_model([1, 50, 100, 1])
        model = lstm.build_model([1, self.seq_len, self.seq_len*2, 1], self.drpt, self.lrate, self.activacion, self.optimizacion, self.perdida)
        model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.seq_len,
            nb_epoch=self.epochs,
            validation_split=0.05)
        predicted = lstm.predict_point_by_point(model, self.X_test)

        #############################################################
        #Metricas
        verdadero = list(map(float, self.y_test))
        predicho = list(map(float, predicted))
    
        MAE = mean_absolute_error(verdadero, predicho)
        MSE = mean_squared_error(verdadero, predicho)
        RMSE = sqrt(MAE)
        R2 = r2_score(verdadero, predicho)

        #print(MAE)
        #print(MSE)
        #print(RMSE)
        #print(R2)
        #print(MAPE)
        ##############################################################
        #Guardado NETCDF
        fecha = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

        nomb = 'Project/data/netCDF4/l96Ipred/L96Ipred-' + fecha + '.nc'

        f = nc4.Dataset(nomb, 'w', format='NETCDF4')
        tempgrp = f.createGroup('l96Ipred')

        tempgrp.createDimension('z', None)

        #Metadatos del dataset
        tempgrp.nombre = "Lorenz 96 I pred."
        tempgrp.fecha = fecha
        tempgrp.dset = self.name
        tempgrp.Factivacion = self.activacion
        tempgrp.Foptimizacion = self.optimizacion
        tempgrp.Fperdida = self.perdida
        tempgrp.MAE = MAE
        tempgrp.MSE = MSE
        tempgrp.RMSE = RMSE
        tempgrp.R2 = R2
 
        Observado = tempgrp.createVariable('Observado', 'f4', 'z')
        Predicho = tempgrp.createVariable('Predicho', 'f4', 'z')
        Tiempo = tempgrp.createVariable('Tiempo', 'f4', 'z')

        tiempo = np.arange(0, len(self.y_test), 1)

        Observado[:] = self.y_test
        Predicho[:] = predicted
        Tiempo[:] = tiempo

        f.close()
        ##############################################################
        #Preparado de datos para ser enviados via ajax
        time = np.arange(0, len(self.y_test), 1)
        predicho = np.array((time, predicted)).T
        verdadero = np.array((time, self.y_test)).T
        resultado = np.array((time, predicted, self.y_test)).T

        return (predicho.tolist(), verdadero.tolist(), resultado.tolist())