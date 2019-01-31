import csv
import numpy as np
import netCDF4 as nc4
from time import gmtime, strftime

class Lorenz63():
    """docstring for Lorenz96II"""
    def __init__(self, sigma, rho, beta, obs, guardar):
     
        self.dt = 0.01 
        self.stepCnt = obs
        self.guardar = guardar
        self.beta = beta
        self.sigma = sigma
        self.rho = rho 

        # Need one more for the initial values
        self.xs = np.empty((self.stepCnt,))
        self.ys = np.empty((self.stepCnt,))
        self.zs = np.empty((self.stepCnt,))

        # Time vector
        self.time = np.arange(0, self.stepCnt, 1)
        
    #def lorenz(self, x, y, z, s=10, r=28, b=2.667):
    def lorenz(self, x, y, z, s, r, b):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
    
        return x_dot, y_dot, z_dot

    def format(x):
        return float("{0:.3f}".format(x))

    def l63(self, poit_pass):

        #Vector auxiliar de tiempo total
        self.timetot = np.arange(0.0, (self.stepCnt+poit_pass), 1)

        if self.guardar == '1':
            fecha = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
        
            #netCDF
            name = 'Project/data/netCDF4/l63/L63-' + fecha + '.nc'
            f = nc4.Dataset(name, 'w', format='NETCDF4')
            tempgrp = f.createGroup('l63')

            tempgrp.createDimension('t', None)
            tempgrp.createDimension('tt', None)

            #Metadatos del dataset
            tempgrp.nombre = "Lorenz 63"
            tempgrp.beta = self.beta
            tempgrp.sigma = self.sigma
            tempgrp.rho = self.rho
            tempgrp.fecha = fecha
            tempgrp.graficar = self.stepCnt
            tempgrp.todos = (self.stepCnt + poit_pass)

            #Creamos las variables 
            tiempo = tempgrp.createVariable('Tiempo', 'f4', 't')
            puntosX = tempgrp.createVariable('PuntosX', 'f4', 't')
            puntosY = tempgrp.createVariable('PuntosY', 'f4', 't')
            puntosZ = tempgrp.createVariable('PuntosZ', 'f4', 't')
            tiempoT = tempgrp.createVariable('TTotal', 'f4', 'tt')
            puntosTX = tempgrp.createVariable('PuntosTX', 'f4', 'tt')
            puntosTY = tempgrp.createVariable('PuntosTY', 'f4', 'tt')
            puntosTZ = tempgrp.createVariable('PuntosTZ', 'f4', 'tt')

        # Variables auxiliares
        self.stepTot = self.stepCnt + poit_pass

        xsAux = np.empty((self.stepTot,))
        ysAux = np.empty((self.stepTot,))
        zsAux = np.empty((self.stepTot,))

        xsAux[0], ysAux[0], zsAux[0] = (0., 1., 1.05)
        auxCont = 0
        
        #Puntos a graficar. 
        for i in range(self.stepTot-1):
            # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = self.lorenz(xsAux[i], ysAux[i], zsAux[i], self.sigma, self.rho, self.beta)
            xsAux[i + 1] = xsAux[i] + (x_dot * self.dt)
            ysAux[i + 1] = ysAux[i] + (y_dot * self.dt)
            zsAux[i + 1] = zsAux[i] + (z_dot * self.dt)
            if i >= (poit_pass-1):
                self.xs[auxCont] = xsAux[i]
                self.ys[auxCont] = ysAux[i]
                self.zs[auxCont] = zsAux[i]
                auxCont += 1

        if self.guardar == '1':
            #NETCDF
            tiempo[:] = self.time
            tiempoT[:] = self.timetot
            puntosX[:] = self.xs
            puntosTX[:] = xsAux
            puntosY[:] = self.ys
            puntosTY[:] = ysAux
            puntosZ[:] = self.zs
            puntosTZ[:] = zsAux

            f.close()

        xx = np.array((self.time, self.xs)).T
        yy = np.array((self.time, self.ys)).T
        zz = np.array((self.time, self.zs)).T
        xyz = np.array((self.time, self.xs, self.ys, self.zs)).T

        return(xx.tolist(), yy.tolist(), zz.tolist(), xyz.tolist())