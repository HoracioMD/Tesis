from . import lstm63
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
import netCDF4 as nc4
from time import gmtime, strftime
from math import sqrt

class PredictL63():

    def __init__(self, name, epochs=15, seq_len=256, dropout=0.2, lrate=0.001, activar='linear', optimizar='adam', perdida='mse'):
        #print(data)

        f = nc4.Dataset('Project/rnn/multivariable/Parametros63.nc', 'r')
        tempgrp = f.groups['l63param']

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

        f.close()

        #self.X_train, self.y_train, self.X_test, self.y_test = lstm.load_data('mycsv.csv', self.seq_len, False)       
        self.train_X, self.train_y, self.test_X, self.test_y = lstm63.load_data(self.name, self.seq_len, False)
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
        model = lstm63.build_model(self.train_X, self.drpt, self.lrate, self.activacion, self.optimizacion, self.perdida)
        # fit network
        history = model.fit(self.train_X, self.train_y, epochs=self.epochs, batch_size=self.seq_len, validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)
        # Prediction
        predicted = lstm63.predict_point_by_point(model, self.test_X)

        # Funcion costo inicial y final 
        costo = history.history['loss']
        #print(history.history['loss'])
        for i in range (len(costo)):
            costoinicial = costo[0]
            if i == (len(costo)-1):
                costofinal = costo[i]

        mejora = ((costoinicial-costofinal)/costoinicial)*100

        #print(costoinicial)
        #print(costofinal)
        #print(mejora)

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

        nomb = 'Project/data/netCDF4/l63pred/L63pred-' + fecha + '.nc'

        f = nc4.Dataset(nomb, 'w', format='NETCDF4')
        tempgrp = f.createGroup('l63pred')

        tempgrp.createDimension('z', None)

        #Metadatos del dataset
        tempgrp.nombre = "Lorenz 63 pred."
        tempgrp.fecha = fecha
        tempgrp.dset = self.name
        tempgrp.Factivacion = self.activacion
        tempgrp.Foptimizacion = self.optimizacion
        tempgrp.Fperdida = self.perdida
        tempgrp.MAE = MAE
        tempgrp.MSE = MSE
        tempgrp.RMSE = RMSE
        tempgrp.R2 = R2
 
        ObservadoX = tempgrp.createVariable('ObservadoX', 'f4', 'z')
        PredichoX = tempgrp.createVariable('PredichoX', 'f4', 'z')
        ObservadoY = tempgrp.createVariable('ObservadoY', 'f4', 'z')
        PredichoY = tempgrp.createVariable('PredichoY', 'f4', 'z')
        ObservadoZ = tempgrp.createVariable('ObservadoZ', 'f4', 'z')
        PredichoZ = tempgrp.createVariable('PredichoZ', 'f4', 'z')
        Tiempo = tempgrp.createVariable('Tiempo', 'f4', 'z')

        tiempo = np.arange(0, len(self.test_y), 1)

        ObservadoX[:] = [x[0] for x in self.test_y]
        PredichoX[:] = [x[0] for x in predicted]
        ObservadoY[:] = [y[1] for y in self.test_y]
        PredichoY[:] = [y[1] for y in predicted]
        ObservadoZ[:] = [z[2] for z in self.test_y]
        PredichoZ[:] = [z[2] for z in predicted]
        Tiempo[:] = tiempo

        f.close()
        ##############################################################
        #Preparado de datos para ser enviados via ajax
        ObservadoX = [x[0] for x in self.test_y]
        PredichoX = [x[0] for x in predicted]
        ObservadoY = [y[1] for y in self.test_y]
        PredichoY = [y[1] for y in predicted]
        ObservadoZ = [z[2] for z in self.test_y]
        PredichoZ = [z[2] for z in predicted]
        time = np.arange(0, len(self.test_y), 1)

        AuxPredictedX = np.reshape(PredichoX, (len(PredichoX),))
        AuxTrueX = np.reshape(ObservadoX, (len(ObservadoX),))
        AuxPredictedY = np.reshape(PredichoY, (len(PredichoY),))
        AuxTrueY = np.reshape(ObservadoY, (len(ObservadoY),))
        AuxPredictedZ = np.reshape(PredichoZ, (len(PredichoZ),))
        AuxTrueZ = np.reshape(ObservadoZ, (len(ObservadoZ),))
        
        predichoX = np.array((time[:500], AuxPredictedX[:500])).T
        verdaderoX = np.array((time[:500], AuxTrueX[:500])).T
        predichoY = np.array((time[:500], AuxPredictedY[:500])).T
        verdaderoY = np.array((time[:500], AuxTrueY[:500])).T
        predichoZ = np.array((time[:500], AuxPredictedZ[:500])).T
        verdaderoZ = np.array((time[:500], AuxTrueZ[:500])).T
        #resultado = np.array((time[:500], AuxPredicted[:500], AuxTrue[:500])).T

        return (predichoX.tolist(), verdaderoX.tolist(), predichoY.tolist(), verdaderoY.tolist(), predichoZ.tolist(), verdaderoZ.tolist(), costoinicial.tolist(), costofinal.tolist(), mejora.tolist())