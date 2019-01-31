import numpy as np
import netCDF4 as nc4
import pandas as pd
from time import gmtime, strftime

class Lorenz96I():
	"""docstring for Lorenz96I"""
	def __init__(self, N, F, obs, guardar):
		self.N = N
		self.x = np.zeros(self.N)
		self.x[0] = 1 
		self.tf = 0.05
		self.dt = 0.001
		self.F = F
		self.s = 1/6 * self.dt
		self.z = 0.5 * self.dt
		self.point_plot = obs
		self.deltas = int(self.tf/self.dt)
		self.t = np.arange(0.0, self.point_plot * self.tf, self.tf)
		self.guardar = guardar

	#Format de salida
	def format(self, dec, p_list):
		return [ '%.4f' % elem for elem in p_list ]

	#Funcion Lorenz
	def f(self, x):
		N = self.N
		k = np.zeros(N)
		k[0] = x[N-2] * (x[1] - x[N-3]) - x[0]
		k[1] = x[0] * (x[2] - x[N-2]) - x[1]
		k[N-1] = x[N-2] * (x[1] - x[N-3])  - x[N-1]
		for i in range(2, N-1):
			k[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
		k = k + self.F

		return k

	#Runge Kutta 4
	def rk4(self, x):
	    k1 = self.f(x)
	    k2 = self.f(x+self.z*k1)
	    k3 = self.f(x+self.z*k2)
	    k4 = self.f(x+k3*self.dt)

	    return self.s*(k1+k2*2+k3*2+k4)

	#Cuerpo principal del programa
	def l96I(self, point_pass):

		#Lista auxiliar para guardar todos los nros creados
		auxiliarDat = []

		if self.guardar == '1':
			#Fecha y hora del sistema en el momento
			fecha = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
			#CSV
			aux_csv = np.zeros(point_pass)
			
			#netCDF
			#Vector auxiliar de tiempo total
			self.tt = np.arange(0.0, (self.point_plot+point_pass) * self.tf, self.tf)

			#Ubicacion donde se guardara el archivo NC
			#name = 'L96UE/L96I-' + "-" + fecha + '.nc'
			name = 'Project/data/netCDF4/l96I/L96I-' + fecha + '.nc'
			f = nc4.Dataset(name, 'w', format='NETCDF4')
			tempgrp = f.createGroup('l96I')

			tempgrp.createDimension('x', None)
			tempgrp.createDimension('y', None)
			tempgrp.createDimension('z', None)

			#Metadatos del dataset
			tempgrp.forzado = self.F
			tempgrp.nombre = "Lorenz 96 I"
			tempgrp.fecha = fecha
			tempgrp.nro_var = self.N
			tempgrp.graficar = self.point_plot
			tempgrp.todos = (self.point_plot + point_pass)
 
			puntos = tempgrp.createVariable('Puntos', 'f4', 'y')
			tiempo = tempgrp.createVariable('Tiempo', 'f4', 'x')
			todos = tempgrp.createVariable('Todos', 'f4', 'z')
			tiempotot = tempgrp.createVariable('TTotal', 'f4', 'z')

		xst = np.zeros(self.point_plot)
		for i in range(point_pass):
			for j in range(self.deltas): 
				self.x += self.rk4(self.x)
				#CSV
				if self.guardar == '1':
					aux_csv[i] = self.x[1]
			auxiliarDat.append(self.x[1])

		for i in range(self.point_plot):
			for j in range(self.deltas): 
				self.x += self.rk4(self.x)
			xst[i] = self.x[0]
			auxiliarDat.append(self.x[1])

		#Lista auxiliar transformada en vector para almacenar
		auxiliarDat = np.array(auxiliarDat)

		if self.guardar == '1':

			puntos[:] = xst
			tiempo[:] = self.t
			todos[:] = auxiliarDat
			tiempotot[:] = self.tt

			f.close()

			#CSV
			my_df = pd.DataFrame(aux_csv)
			my_df.to_csv('Project/data/netCDF4/l96I/L96I-' + fecha + '.csv', index=False, header=False)

		z = np.array((self.t, xst)).T 

		#return(z.tolist(), t.tolist(), xst.tolist())
		return(z.tolist())