from . import Auxlstm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
import netCDF4 as nc4
from time import gmtime, strftime
from math import sqrt

class PredictL96I():

    def __init__(self, name, epochs=10, seq_len=200, dropout=0.1, lrate=0.001, activar='linear', optimizar='adam', perdida='mse'):
        #print(data)

        f = nc4.Dataset('Project/rnn/univariable/Parametros96.nc', 'r')
        tempgrp = f.groups['l96param']

        self.name = name
        #self.epochs = epochs
        #self.seq_len = seq_len
        #self.drpt = dropout
        #self.lrate = lrate
        #self.activacion = activar
        #self.optimizacion = optimizar
        #self.perdida = perdida

        self.epochs = tempgrp.epochs
        self.seq_len = tempgrp.seq_len
        self.drpt = tempgrp.drpt
        self.lrate = tempgrp.lrate
        self.activacion = tempgrp.activacion
        self.optimizacion = tempgrp.optimizacion
        self.perdida = tempgrp.perdida

        #self.X_train, self.y_train, self.X_test, self.y_test = lstm.load_data('mycsv.csv', self.seq_len, False)       
        self.train_X, self.train_y, self.test_X, self.test_y = Auxlstm.load_data(self.name, self.seq_len, False)
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
        model = Auxlstm.build_model(self.train_X, self.drpt, self.lrate, self.activacion, self.optimizacion, self.perdida)
        # fit network
        history = model.fit(self.train_X, self.train_y, epochs=self.epochs, batch_size=self.seq_len, validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)
        # Prediction
        predicted = Auxlstm.predict_point_by_point(model, self.test_X)

         # Funcion costo inicial y final 
        costo = history.history['loss']
        #print(history.history['loss'])
        for i in range (len(costo)):
            costoinicial = costo[0]
            if i == (len(costo)-1):
                costofinal = costo[i]

        mejora = ((costoinicial-costofinal)/costoinicial)*100

        #############################################################
        #Metricas
        #verdadero = list(map(float, self.test_y))
        #predicho = list(map(float, predicted))
    
        #MAE = mean_absolute_error(verdadero, predicho)
        #MSE = mean_squared_error(verdadero, predicho)
        #RMSE = sqrt(MAE)
        #R2 = r2_score(verdadero, predicho)

        MAE = mean_absolute_error(self.test_y, predicted)
        MSE = mean_squared_error(self.test_y, predicted)
        RMSE = sqrt(MSE)
        R2 = r2_score(self.test_y, predicted)

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

        tiempo = np.arange(0, len(self.test_y), 1)

        Observado[:] = self.test_y
        Predicho[:] = predicted
        Tiempo[:] = tiempo

        f.close()
        ##############################################################
        #Preparado de datos para ser enviados via ajax
        AuxPredicted = np.reshape(predicted, (len(predicted),))
        AuxTrue = np.reshape(self.test_y, (len(self.test_y),))

        time = np.arange(0, len(self.test_y), 1)
        predicho = np.array((time[:200], AuxPredicted[:200])).T
        verdadero = np.array((time[:200], AuxTrue[:200])).T
        resultado = np.array((time[:200], AuxPredicted[:200], AuxTrue[:200])).T

        return (predicho.tolist(), verdadero.tolist(), resultado.tolist(), costoinicial.tolist(), costofinal.tolist(), mejora.tolist())