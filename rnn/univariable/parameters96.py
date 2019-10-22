import netCDF4 as nc4

class ParametersL96():

    def __init__(self, epochs=15, seq_len=256, dropout=0.2, lrate=0.001, activar='linear', optimizar='adam', perdida='mse'):
        #print(data)
        self.epochs = epochs
        self.seq_len = seq_len
        self.drpt = dropout
        self.lrate = lrate
        self.activacion = activar
        self.optimizacion = optimizar
        self.perdida = perdida

        f = nc4.Dataset('Project/rnn/univariable/Parametros96.nc', 'w', format='NETCDF4')
        tempgrp = f.createGroup('l96param')

        #Metadatos del dataset
        tempgrp.nombre = "Lorenz 96 parametros"
        tempgrp.epochs = self.epochs
        tempgrp.seq_len = self.seq_len
        tempgrp.drpt = self.drpt
        tempgrp.lrate = self.lrate
        tempgrp.activacion = self.activacion
        tempgrp.optimizacion = self.optimizacion
        tempgrp.perdida = self.perdida

        f.close()
        