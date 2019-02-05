import os
import time
import warnings
import numpy as np
import netCDF4 as nc4
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

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
    
    return (aux)

def load_data(filename, seq_len, normalise_window):
    #f = open(os.path.abspath(os.path.join('/home/Documentos/Codes_Django/Tesis/Project/rnn/univariable/', 'mycsv.csv')), 'rb').read()
    #data = f.decode().split('\n')
    #print(data)
    data = extract_l96I_data(filename)

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    
    print(result.shape)
    print(int(row))
    
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers, drpt, lrate, activacion, optimizacion, perdida):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(drpt))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(drpt))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation(activacion))

    start = time.time()
    model.compile(loss=perdida, optimizer=optimizacion)
    #print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
