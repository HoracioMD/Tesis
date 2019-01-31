import numpy as np
import netCDF4 as nc4
import csv
from time import gmtime, strftime

class Lorenz96II():
    """docstring for Lorenz96II"""
    def __init__(self, N, M, F, obs, guardar):
        self.N = N
        self.M = M
        #self.J = self.M / self.N
        self.J = 32
        self.h = 1
        self.c = 10
        self.b = 10
        self.cb = 10*10

        self.x = np.zeros(self.N)
        self.x[0] = 1
        self.y = np.zeros(self.M)
        self.y[0] = 1

        self.tf = 0.05
        self.dt = 0.001
        self.F = F
        self.s = 1./6
        self.z = 0.5
        self.point_plot = obs
        self.deltas = int(self.tf/self.dt)
        self.t = np.arange(0.0, self.point_plot * self.tf, self.tf)
        self.guardar = guardar

        self.const = ((1*10)/10) #constante resultado de la operacion ((h*c)/b)

        self.auxVX = []


    def format(self, dec, p_list):
        return [ '%.4f' % elem for elem in p_list ]

    def Lorenz96x(self, x, F, AuxV):
        N = self.N
        vx = np.zeros(N)

        vx[0] = (x[1] - x[N-3]) * x[N-2] - x[0]
        vx[1] = (x[2] - x[N-2]) * x[0]  - x[1]
        vx[N-1] = (x[1] - x[N-3]) * x[N-2]  - x[N-1]
    
        for i in range(2, N-1):
            vx[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        vx = vx + F - (self.const*AuxV)
    
        return vx

    def Lorenz96y(self, y, x, AuxV):
        M = self.M
        vy = np.zeros(M)
    
        vy[0] = ((self.cb*y[1]) * (y[M-2] - y[2]) - self.c*y[0]) + (self.const * x[0])
        vy[M-2] = ((self.cb*y[M-1]) * (y[M-3] - y[1]) - self.c*y[M-2]) + (self.const * x[7])
        vy[M-1] = ((self.cb*y[1]) * (y[M-2] - y[2])  - self.c*y[M-1]) + (self.const * x[7])
    
        for i in range(1, M-2):
            vy[i] =  (self.cb*y[i+1]) * (y[i-1] - y[i+2]) - self.c*y[i] + (self.const * x[int(i/32)])
        
        return vy

    def RungeKuttaXY(self, xy, Fi, Fxy, Vec):
    
        k1xy = Fxy(xy, Fi, Vec)
        k2xy = Fxy(xy+self.z*k1xy*self.dt, Fi, Vec)
        k3xy = Fxy(xy+self.z*k2xy*self.dt, Fi, Vec)
        k4xy = Fxy(xy+k3xy*self.dt, Fi, Vec)

        RK4xy = self.s*self.dt*(k1xy+k2xy*2+k3xy*2+k4xy)
        xy += RK4xy
    
        return xy

    def sumatoria(self, vector):
        N = self.N
        acum = np.zeros(N)
        for h in range(N):
            auxmin = int(self.J*((h+1)-1))
            auxmax = int((h+1)*self.J)
   
            for i in range(auxmin, auxmax):
                acum[h] = acum[h] + vector[i]
        return acum

    def l96II(self, point_pass):

        #Lista auxiliar para guardar todos los nros creados
        auxiliarDatX = []
        auxiliarDatY = []

        #Vector auxiliar de tiempo total
        self.tt = np.arange(0.0, (self.point_plot+point_pass) * self.tf, self.tf)

        if self.guardar == '1':
            fecha = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
            #tsave = np.arange(0.0, point_pass * self.tf, self.tf)
            #netCDF
            name = 'Project/data/netCDF4/l96II/L96II-' + fecha + '.nc'
            f = nc4.Dataset(name, 'w', format='NETCDF4')
            tempgrp = f.createGroup('l96II')

            #Configuramos las dimensiones del archivo
            tempgrp.createDimension('t', None)
            tempgrp.createDimension('tt', None)

            #Metadatos del dataset
            tempgrp.nombre = "Lorenz 96 II"
            tempgrp.varX = self.N
            tempgrp.varY = self.M
            tempgrp.F = self.F
            tempgrp.fecha = fecha
            tempgrp.graficar = self.point_plot
            tempgrp.todos = (self.point_plot + point_pass)

            #Creamos las variables 
            tiempo = tempgrp.createVariable('Tiempo', 'f4', 't')
            puntosX = tempgrp.createVariable('PuntosX', 'f4', 't')
            puntosY = tempgrp.createVariable('PuntosY', 'f4', 't')
            tiempoT = tempgrp.createVariable('TTotal', 'f4', 'tt')
            puntosTX = tempgrp.createVariable('PuntosTX', 'f4', 'tt')
            puntosTY = tempgrp.createVariable('PuntosTY', 'f4', 'tt')

        #Cuerpo principal
        # Variables auxiliares
        aux1 = np.zeros(self.point_plot)
        aux2 = np.zeros(self.point_plot)
        auxVX = self.auxVX

        for i in range(point_pass):
            auxVX = self.sumatoria(self.y)
            for n in range(self.deltas):
                TsY = self.RungeKuttaXY(self.y, self.x, self.Lorenz96y, auxVX)
                TsX = self.RungeKuttaXY(self.x, self.F, self.Lorenz96x, auxVX)
            self.y = TsY
            self.x = TsX
            auxiliarDatX.append(self.x[1])
            auxiliarDatY.append(self.x[1])

        for i in range(self.point_plot):
            auxVX = self.sumatoria(self.y)
            for n in range (self.deltas):
                TsY = self.RungeKuttaXY(self.y, self.x, self.Lorenz96y, auxVX)    
                TsX = self.RungeKuttaXY(self.x, self.F, self.Lorenz96x, auxVX)
            self.y = TsY
            self.x = TsX
            aux1[i] = self.x[1]
            aux2[i] = self.y[1]
            auxiliarDatX.append(self.x[1])
            auxiliarDatY.append(self.x[1])

        #Lista auxiliar transformada en vector para almacenar
        auxiliarDatX = np.array(auxiliarDatX)
        auxiliarDatY = np.array(auxiliarDatY)

        #Guardar datos archivo nc
        if self.guardar == '1':
            tiempo[:] = self.t
            tiempoT[:] = self.tt
            puntosX[:] = aux1
            puntosTX[:] = auxiliarDatX
            puntosY[:] = aux2
            puntosTY[:] = auxiliarDatY

            f.close()

        zx = np.array((self.t, aux1)).T
        zy = np.array((self.t, aux2)).T
        zxy = np.array((self.t, aux1, aux2)).T

        return(zx.tolist(), zy.tolist(), zxy.tolist())
