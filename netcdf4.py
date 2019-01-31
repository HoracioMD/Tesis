import netCDF4 as nc4
import numpy as np
import os
from datetime import datetime

class NCDF4():

    def __init__(self):
        self.dir_base = 'data/netCDF4/'

    def return_dataset(self, dir, action):
        return nc4.Dataset(os.path.join(os.path.dirname(__file__),dir), action, format='NETCDF4')

    def extract_l96I_data(self, filename):
        dir = self.dir_base + "l96I/" + str(filename) + ".nc"
        directory = os.fsencode(os.path.join(os.path.dirname(__file__),dir))
        dataset_nc4 = self.return_dataset(str(directory, 'utf-8'), 'r')
        group = dataset_nc4.groups['l96I']
        x = group.variables['Puntos'][:]
        t = group.variables['Tiempo'][:]
        z = np.array((t, x)).T 

        return (z.tolist())

    def extract_l96II_data(self, filename):
        dir = self.dir_base + "l96II/" + str(filename) + ".nc"
        directory = os.fsencode(os.path.join(os.path.dirname(__file__),dir))
        dataset_nc4 = self.return_dataset(str(directory, 'utf-8'), 'r')
        group = dataset_nc4.groups['l96II']
        x = group.variables['PuntosX'][:]
        y = group.variables['PuntosY'][:]
        t = group.variables['Tiempo'][:]
        
        zx = np.array((t, x)).T
        zy = np.array((t, y)).T 

        return (zx.tolist(), zy.tolist())

    def extract_l63_data(self, filename):
        dir = self.dir_base + "l63/" + str(filename) + ".nc"
        directory = os.fsencode(os.path.join(os.path.dirname(__file__),dir))
        dataset_nc4 = self.return_dataset(str(directory, 'utf-8'), 'r')
        group = dataset_nc4.groups['l63']
        x = group.variables['PuntosX'][:]
        y = group.variables['PuntosY'][:]
        z = group.variables['PuntosZ'][:]
        t = group.variables['Tiempo'][:] 

        xx = np.array((t, x)).T
        yy = np.array((t, y)).T
        zz = np.array((t, z)).T
        #xyz = np.array((t, x, y, z)).T

        return (xx.tolist(), yy.tolist(), zz.tolist())
        #return (xx.tolist(), yy.tolist(), zz.tolist(), xyz.tolist())

    def return_list_nc(self, model):
        list_files = {} 
        dir = self.dir_base + str(model) + "/"
        directory = os.fsencode(os.path.join(os.path.dirname(__file__),dir))
        #print(str(directory))
        for file in self._iterate_folder(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".nc"):
                dataset = self.return_dataset(str(directory, 'utf-8')+str(filename), 'r')
                name = filename.split(".",1)[0]
                grp = dataset.groups[str(model)]
                list_files[str(name)] = {
                    'nombre': grp.nombre,
                    'forzado':grp.forzado,
                    'fecha':grp.fecha,
                    'n':grp.nro_var,
                    'obs':grp.graficar,
                    'longitud': grp.todos
                }
        return list_files

    def return_list63_nc(self, model):
        list_files = {} 
        dir = self.dir_base + str(model) + "/"
        directory = os.fsencode(os.path.join(os.path.dirname(__file__),dir))
        #print(str(directory))
        for file in self._iterate_folder(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".nc"):
                dataset = self.return_dataset(str(directory, 'utf-8')+str(filename), 'r')
                name = filename.split(".",1)[0]
                grp = dataset.groups[str(model)]
                list_files[str(name)] = {
                    'nombre': grp.nombre,
                    'fecha': grp.fecha,
                    'beta': grp.beta,
                    'sigma' : grp.sigma,
                    'rho' : grp.rho,
                    'obs': grp.graficar,
                    'longitud': grp.todos
                }
        return list_files

    def return_list2_nc(self, model):
        list_files = {} 
        dir = self.dir_base + str(model) + "/"
        directory = os.fsencode(os.path.join(os.path.dirname(__file__),dir))
        #print(str(directory))
        for file in self._iterate_folder(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".nc"):
                dataset = self.return_dataset(str(directory, 'utf-8')+str(filename), 'r')
                name = filename.split(".",1)[0]
                grp = dataset.groups[str(model)]
                list_files[str(name)] = {
                    'nombre': grp.nombre,
                    'fecha': grp.fecha,
                    'forzado': grp.F,
                    'varX' : grp.varX,
                    'varY' : grp.varY,
                    'obs': grp.graficar,
                    'longitud': grp.todos
                }
        return list_files

    def _iterate_folder(self, directory):
        for file in os.listdir(directory):
            yield file