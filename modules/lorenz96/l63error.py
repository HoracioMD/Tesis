import csv
import numpy as np
import numpy.random as rnd
import pandas as pd

class Lorenz63error():
    """docstring for Lorenz96II"""
    def __init__(self, sigma, rho, beta, errorxy, errorz, obs):
     
        self.dt = 0.01 
        self.stepCnt = obs-1
        self.beta = beta
        self.sigma = sigma
        self.rho = rho 
        self.XYAux = errorxy
        self.ZAux = errorz

        # Need one more for the initial values
        self.xs = np.empty((self.stepCnt +1,))
        self.ys = np.empty((self.stepCnt +1,))
        self.zs = np.empty((self.stepCnt +1,))

        # Setting initial values
        self.xs[0], self.ys[0], self.zs[0] = (17.36144399, 11.64080574, 46.40741375)

        # Time vector
        self.time = np.arange(0, 1000, 1)
        
    #def lorenz(self, x, y, z, s=10, r=28, b=2.667):
    def lorenz(self, x, y, z, s, r, b):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
    
        return x_dot, y_dot, z_dot

    def format(x):
        return float("{0:.3f}".format(x))

    def generateError(self,vecX,vecY,vecZ):
        #Numeros randoms para los errores
        #XYAux = 3
        #ZAux = 6
        xst = []
        yst = []
        zst = []

        aux1 = rnd.normal(0, 1, self.stepCnt+1)
        aux2 = rnd.normal(0, 1, self.stepCnt+1)
        aux3 = rnd.normal(0, 1, self.stepCnt+1)

        xst[:] = vecX[:] + np.dot(self.XYAux,aux1)
        yst[:] = vecY[:] + np.dot(self.XYAux,aux2)
        zst[:] = vecZ[:] + np.dot(self.ZAux,aux3)

        return xst,yst,zst

    def saveResults(self,xs,ys,zs,xst,yst,zst):
        valores = np.array((xs, ys, zs)).T
        valores2 = np.array((xst, yst, zst)).T    

        my_df = pd.DataFrame(valores)
        my_df.to_csv('Project/data/netCDF4/ass/Datos.csv', index=False, header=False)

        my_df2 = pd.DataFrame(valores2)
        my_df2.to_csv('Project/data/netCDF4/ass/Errores.csv', index=False, header=False)

    def l63error(self):

        self.poit_pass = 1000

        xsAux = np.empty((self.stepCnt+1,))
        ysAux = np.empty((self.stepCnt+1,))
        zsAux = np.empty((self.stepCnt+1,))
        xst = []
        yst = []
        zst = []

        xsAux[0], ysAux[0], zsAux[0] = (17.36144399, 11.64080574, 46.40741375)
        auxCont = 0
        
        #Puntos a graficar. 
        for i in range(self.stepCnt):
            # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = self.lorenz(xsAux[i], ysAux[i], zsAux[i], self.sigma, self.rho, self.beta)
            xsAux[i + 1] = xsAux[i] + (x_dot * self.dt)
            ysAux[i + 1] = ysAux[i] + (y_dot * self.dt)
            zsAux[i + 1] = zsAux[i] + (z_dot * self.dt)

        xst,yst,zst = self.generateError(xsAux,ysAux,zsAux)
        self.saveResults(xsAux,ysAux,zsAux,xst,yst,zst)

        #1000 puntos pasados para graficar
        xx = np.array((self.time, xsAux[:1000])).T
        yy = np.array((self.time, ysAux[:1000])).T
        zz = np.array((self.time, zsAux[:1000])).T
        xxe = np.array((self.time, xst[:1000])).T
        yye = np.array((self.time, yst[:1000])).T
        zze = np.array((self.time, zst[:1000])).T

        return(xx.tolist(), yy.tolist(), zz.tolist(), xxe.tolist(), yye.tolist(), zze.tolist())