import os
import numpy as np
import netCDF4 as nc4
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM

def return_dataset(dir, action):
    return nc4.Dataset(os.path.join(os.path.dirname(__file__),dir), action, format='NETCDF4')

def extract_l96I_data(filename):
    aux = '/home/Documentos/Codes_Django/Tesis/Project/'
    dir = 'data/netCDF4/' + "l96I/" + str(filename) + ".nc"
    #directory = os.fsencode(os.path.join(os.path.dirname(__file__),dir))
    directory = os.fsencode(os.path.join(aux,dir)) 
    #print(directory)
    dataset_nc4 = return_dataset(str(directory, 'utf-8'), 'r')
    group = dataset_nc4.groups['l96I']
    x = group.variables['Todos'][:]
    x1 = x.tolist()
    aux = ['{:.10f}'.format(i) for i in x1]

    newlist=[]
    for item in aux:
        newlist.append(item.split(' '))

    newlist2=pd.DataFrame(newlist,columns=["date"])[1:]
    
    return (newlist2.values.astype(float))

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
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

def load_data(filename, seq_len, normalise_window):
    #f = open(os.path.abspath(os.path.join('/home/Documentos/Codes_Django/Tesis/Project/rnn/univariable/', 'mycsv.csv')), 'rb').read()
    #data = f.decode().split('\n')
    #print(data)
    values = extract_l96I_data(filename)

    row = round(0.85 * values.shape[0])
    #row = 148500

    raw = pd.DataFrame()
    raw['ob1'] = [x[0] for x in values]
    values = raw.values
    valX, valY = split_sequences(values, 50)

    # split into input and outputs
    train_X, train_y = valX[:row, :], valY[:row]
    test_X, test_y = valX[row:, :], valY[row:]

    return [train_X, train_y, test_X, test_y]

def build_model(layers, drpt, lrate, activacion, optimizacion, perdida):
    model = Sequential()
    model.add(LSTM(50, input_shape=(layers.shape[1], layers.shape[2])))
    model.add(Dropout(drpt))
    model.add(Dense(1))
    model.add(Activation(activacion))
    model.compile(loss=perdida, optimizer=optimizacion)

    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    # make a prediction
    ypred = model.predict(data)
    
    return ypred