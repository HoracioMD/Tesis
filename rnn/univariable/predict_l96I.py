from . import lstm
import time
import matplotlib.pyplot as plt
import numpy as np

class PredictL96I():

    def __init__(self, epochs, seq_len, data):
        #print(data)
        self.epochs = epochs
        self.seq_len = seq_len
        #self.X_train, self.y_train, self.X_test, self.y_test = lstm.load_data('mycsv.csv', self.seq_len, False)       
        self.X_train, self.y_train, self.X_test, self.y_test = lstm.load_data('mycsv.csv', self.seq_len, False)
        # Soluciona el problema de hilos feed dict lstm_:0 etc
        import keras.backend.tensorflow_backend
        if keras.backend.tensorflow_backend._SESSION:
           import tensorflow as tf
           tf.reset_default_graph()
           keras.backend.tensorflow_backend._SESSION.close()
           keras.backend.tensorflow_backend._SESSION = None

    def start_prediction(self):
        #model = lstm.build_model([1, 50, 100, 1])
        model = lstm.build_model([1, self.seq_len, self.seq_len*2, 1])
        model.fit(
            self.X_train,
            self.y_train,
            batch_size=512,
            nb_epoch=self.epochs,
            validation_split=0.05)
        predicted = lstm.predict_point_by_point(model, self.X_test)

        time = np.arange(0, len(self.y_test), 1)
        predicho = np.array((time, predicted)).T
        verdadero = np.array((time, self.y_test)).T
        resultado = np.array((time, predicted, self.y_test)).T

        return (predicho.tolist(), verdadero.tolist(), resultado.tolist())

