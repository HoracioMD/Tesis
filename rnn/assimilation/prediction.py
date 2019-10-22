from scipy import stats
from math import sqrt
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam,RMSprop,SGD
import numpy as np
from numpy import random as rnd
import tensorflow as tf
    
class prediction():
    def __init__(self):

        import keras.backend.tensorflow_backend

        if keras.backend.tensorflow_backend._SESSION:
            #import tensorflow as tf
            tf.reset_default_graph()
            keras.backend.tensorflow_backend._SESSION.close()
            keras.backend.tensorflow_backend._SESSION = None

    # Separar los valores predichos y verderos para graficar
    def split_values(self,sequences):
        x, y, z = list(), list(), list()
        for i in range(len(sequences)):
            seq_x = sequences[i][0]
            seq_y = sequences[i][1]
            seq_z = sequences[i][2]
            x.append(seq_x)
            y.append(seq_y)
            z.append(seq_z)
        
        return np.array(x), np.array(y), np.array(z)

    # split a multivariate sequence into samples
    def split_sequences(self,sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
                # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix], sequences[end_ix-1]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)

    def savemodel(self,model):
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("Project/data/netCDF4/ass/model.h5")

    def loadmodel():
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
    
        return loaded_model

    def dataload(self,name):
        # load dataset
        #scaler = StandardScaler()
        dataset = read_csv(name)
        values = dataset.values
    
        #scaler.fit(values)
        #values = scaler.transform(values)
        row = round(0.80 * values.shape[0])
        #row = 148500

        raw = DataFrame()
        raw['ob1'] = [x[0] for x in values]
        raw['ob2'] = [y[1] for y in values]
        raw['ob3'] = [z[2] for z in values]
        values = raw.values
        valX, valY = self.split_sequences(values, 50)

        # split into input and outputs
        train_X, train_y = valX[:row, :], valY[:row]
        test_X, test_y = valX[row:, :], valY[row:]
        #print(test_X.shape)
        # reshape input to be 3D [samples, timesteps, features]
        #print("Samples --- Time Step --- Features per samples (dimension)")
        #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
        return train_X, train_y, test_X, test_y

    def configuremodel(self,train_X):
        #design network
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(3))
        model.add(Activation('linear'))
        model.compile(loss='mae', optimizer='adam')
    
        return model

    def startPred(self):
        # main
        # load data
        train_X, train_y, test_X, test_y = self.dataload('Project/data/netCDF4/ass/Errores.csv')
        train_XX, train_yy, test_XX, test_yy = self.dataload('Project/data/netCDF4/ass/Datos.csv')

        # configure model
        model = self.configuremodel(train_X)
        #fit network
        history = model.fit(train_X, train_y, epochs=15, batch_size=256, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        #save model
        self.savemodel(model)

        # load model
        #model = loadmodel()

        # make a prediction
        ypred = model.predict(test_XX)

        # calculate RMSE
        Totrmse = sqrt(mean_squared_error(test_yy, ypred))
        print('Test RMSE: %.3f' % Totrmse)

        #split values for multi-plot
        trueX, trueY, trueZ = self.split_values(test_yy)
        predX, predY, predZ = self.split_values(ypred)

        #rmse for each variable
        rmseX = sqrt(mean_squared_error(trueX, predX))
        rmseY = sqrt(mean_squared_error(trueY, predY))
        rmseZ = sqrt(mean_squared_error(trueZ, predZ))

        tiempo = np.arange(0, 1000, 1)

        trueX = np.array((tiempo, trueX[:1000])).T
        trueY = np.array((tiempo, trueY[:1000])).T
        trueZ = np.array((tiempo, trueZ[:1000])).T
        predX = np.array((tiempo, predX[:1000])).T
        predY = np.array((tiempo, predY[:1000])).T
        predZ = np.array((tiempo, predZ[:1000])).T

        return (trueX.tolist(), trueY.tolist(), trueZ.tolist(), predX.tolist(), predY.tolist(), predZ.tolist())