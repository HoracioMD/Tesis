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

import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rnd
from numpy.linalg import pinv as pinv

class assimilation():
    def __init__(self):
        #####Variables#####
        self.auxiliarDatos = np.zeros((30,1000,3))
        self.data = []
        self.perturb = []
        self.errors = 'Project/data/netCDF4/ass/Errores.csv'
        self.datas = 'Project/data/netCDF4/ass/Datos.csv'
        self.nombre = 'Project/data/netCDF4/ass/model.h5'

        #####Load data from dataset#####
        self.train_X, self.train_y, self.test_X, self.test_y = self.dataload(self.errors)
        self.train_XX, self.train_yy, self.test_XX, self.test_yy = self.dataload(self.datas)

        #####Take first value from test group (First value after training lot)#####
        self.data.append(self.test_XX[0][0])
        self.data = np.array(self.data)


    # split a multivariate sequence into samples
    def split_sequences(self, sequences, n_steps):
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

    def dataload(self,name):
        # load dataset
        #scaler = StandardScaler()
        dataset = read_csv(name)
        values = dataset.values
        #scaler.fit(values)
        #values = scaler.transform(values)
        row = round(0.80 * values.shape[0])

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

    def loadmodel(self, name):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(name)
                
        return loaded_model

    def lorenz(self,x, y, z, s=10., r=28., b=8./3.):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
                
        return x_dot, y_dot, z_dot

    def evolutionLorenz(self,data):
        dt = 0.01
        stepCnt = 49

        # Need one more for the initial values
        xs = np.empty((50,))
        ys = np.empty((50,))
        zs = np.empty((50,))
            
        # Setting initial values
        #xs[0], ys[0], zs[0] = (data[0][0], data[0][1], data[0][2])
        xs[0], ys[0], zs[0] = (data[0], data[1], data[2])
            
        # Stepping through "time".
        for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
            x_dot, y_dot, z_dot = self.lorenz(xs[i], ys[i], zs[i])
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)
                
        todos = np.array((xs, ys, zs)).T     
            
        return todos

    def perturbations(self,data, perturb):
        Aux = 1
        aux1 = rnd.normal(0, 1, 30)

        #Generando las 30 perturbaciones
        for i in range(30):
            xst = data + np.dot(Aux,aux1[i])
            perturb.append(xst[0])
                
        #Solo las 30 perturbaciones
        perturb = np.array(perturb)
            
        #return data, data2
        return perturb

    def prepareData(self,t51list):
        auxX = [] #X values [30 predictions]
        auxY = [] #Y values [30 predictions]
        auxZ = [] #Z values [30 predictions]
        Xf = []
        for i in range(30):
            auxX.append(t51list[i][0][0])
            auxY.append(t51list[i][0][1])
            auxZ.append(t51list[i][0][2])
                
        auxX = np.array(auxX)
        auxY = np.array(auxY)
        auxZ = np.array(auxZ)

        Xf.append(auxX)
        Xf.append(auxY)
        Xf.append(auxZ)
            
        return (np.array(Xf))

    def concatenate(self,listado,assT51):
            
        zeros = np.zeros((30,3))
            
        for i in range(30):
            zeros[i][0] = assT51[0][i]
            zeros[i][1] = assT51[1][i]
            zeros[i][2] = assT51[2][i]
                
        auxiliar = listado

        for i in range (30):
            for j in range(50):
                if j < 49:
                    auxiliar[i][j]=auxiliar[i][j+1]
                elif j == 49:
                    auxiliar[i][j]=zeros[i]
            
        return auxiliar

    #Codigos del profe
    def sqrt_eig(self,A):
        #"Compute square root using eigenvectors"

        s, V = np.linalg.eigh(A)#assume simmetry of A

        n=V.shape[1]
        m=V.shape[0]

        sqrts = np.sqrt(s)
        n = np.size(s)
        sqrtS = np.zeros((n,n))
        sqrtS[:n, :n] = np.diag(sqrts)

        sqrtA=np.dot(V,np.dot(sqrtS, V.T))

        return sqrtA

    def perobs(self,Xf,yo,H,sqR):
        #xf matriz de 2 dimensiones (estados, 30 miembros) (30 estados 51) [[30 estados X][30 estados Y][30 estados Z]]
        #yo observacion en el tiempo especifico (observacion 51) [X obs, Y obs, Z obs]
            
        #" Assimilate using classical enkf. Pertubed observations"
                
        #print(Xf)
        #print(yo)
                
        [nx,nem]=Xf.shape
        ny=yo.shape[0]
        finf = 1.5 #1.1 o se puede achicar a 1 o 1.01

        #print(H)
        #print(sqR)

        if finf > 1:
            xfm= np.reshape(np.mean(Xf,1),[nx,1])
            Xf=xfm+(finf)**0.5*(Xf-xfm)
                
        Pfxx= np.cov(Xf)

        # Classical Kalman gain
        K = Pfxx.dot(H.T).dot(pinv(H.dot(Pfxx).dot(H.T) + sqR.dot(sqR.T)))

        wrk=rnd.normal(0, 1, [ny,nem])
        Y=np.dot(H,Xf)+np.dot(sqR,wrk)    
        innov = np.reshape(yo,[ny,1]) - Y

        Xa = Xf + np.dot(K,innov)
                
        #print(Xa[2])

        return Xa
            
    def initialization(self):
        Rfac = 0.5
        Qfac = 0.1 #Lo puedo achicar a 0.2 o 0.1

        nx=3
                
        # model error
        Q = np.eye(nx) * Qfac
        sqQ = self.sqrt_eig(Q)
                
        # Usa este
        # Error en las observaciones
        ny= 3
        R = np.eye(ny) * Rfac
        sqR = self.sqrt_eig(R)

        # observation operator
        H = np.eye(ny)

        #print(sqR)
        #print(H)
                
        return H,sqR

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
            
    def plot_results_ass(self,trueX, trueY, trueZ,listado,rmseX,rmseY,rmseZ,name,name2):    
        fig = plt.figure(figsize=(9,3), edgecolor='black')
        fig.subplots_adjust(bottom=0.18,top=0.90,right=0.98,left=0.08,wspace=0.27,hspace=0.2)
        ax = fig.add_subplot(1,3,1)
        for i in range(30):
            valXass, valYass, valZass = self.split_values(listado[i])
            ax.plot(valXass, '.', color='orange',markersize=3)
        ax.plot(trueX, markersize=3)
        plt.title('X RMSE : %.3f' % rmseX)
        plt.ylabel('Values')
        #plt.legend()
            
        ax = fig.add_subplot(1,3,2)
        for i in range(30):
            valXass, valYass, valZass = self.split_values(listado[i])
            if i == 29:
                ax.plot(valYass, '.', color='orange',markersize=3, label='Ass val')
            else:
                ax.plot(valYass, '.', color='orange',markersize=3)
        ax.plot(trueY, markersize=3, label="True val")
        plt.title('Y RMSE : %.3f' % rmseY)
        plt.xlabel('Points')
        plt.legend()
            
        ax = fig.add_subplot(1,3,3)
        for i in range(30):
            valXass, valYass, valZass = self.split_values(listado[i])
            ax.plot(valZass, '.', color='orange',markersize=3)
        ax.plot(trueZ, markersize=3)
        plt.title('Z RMSE : %.3f' % rmseZ)
        #plt.legend()
            
        plt.show()
        fig.savefig(name)
        fig.savefig(name2)
        plt.close(fig)
            
    def plotAll(self,trueX, obsX, trueY, obsY, trueZ, obsZ):
        fig = plt.figure(figsize=(9,3))
        fig.subplots_adjust(bottom=0.18,top=0.90,right=0.98,left=0.08,wspace=0.27,hspace=0.2)
        ax = fig.add_subplot(1,3,1)
        ax.plot(trueX, markersize=3)
        ax.plot(obsX, '--', markersize=3)
        plt.title('X Points') 
        plt.ylabel('Values')
        #plt.legend()
            
        ax = fig.add_subplot(1,3,2)
        ax.plot(trueY, markersize=3, label="True val")
        ax.plot(obsY, '--', markersize=3, label='Obs val')
        plt.title('Y Points')
        plt.xlabel('Points')
        plt.legend()
            
        ax = fig.add_subplot(1,3,3)
        ax.plot(trueZ, markersize=3)
        ax.plot(obsZ, '--', markersize=3)
        plt.title('Z Points')
        #plt.legend()
            
        plt.show()
        fig.savefig('vals.eps')
        plt.close(fig)
            
    def plots(self,listado,valZtrue):    
        fig = plt.figure(facecolor='white', figsize=(12, 8))
        ax = fig.add_subplot(111)
        for i in range(30):
            valXass, valYass, valZass = self.split_values(listado[i])
            ax.plot(valZass, '.', color='orange')
        plt.plot(valZtrue, label='True Val')
        plt.title('Points')
        plt.xlabel('Points')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    def promedio(self,listado):
        acumX = 0
        acumY = 0
        acumZ = 0
        promX = np.zeros(50)
        promY = np.zeros(50)
        promZ = np.zeros(50)
            
        for j in range(50):
            for i in range(30):
                acumX += listado[i][j][0]
                acumY += listado[i][j][1]
                acumZ += listado[i][j][2]
                    
            acumX = acumX/30
            acumY = acumY/30
            acumZ = acumZ/30
                
            promX[j] = acumX
            promY[j] = acumY
            promZ[j] = acumZ
            
        return promX, promY, promZ
            
    def todosDatos(self,listado,datos,i):
            
        for j in range (30):
            datos[j][i][0] = listado[j][49][0]
            datos[j][i][1] = listado[j][49][1]
            datos[j][i][2] = listado[j][49][2]
            
        return datos

    def main(self):
        ############################ Main ##########################
        #####Variables#####
        
        #auxiliarDatos = np.zeros((30,1000,3))
        #data = []
        #perturb = []
        #name = 'Errores.csv'
        #name2 = 'Datos.csv'

        #####Load data from dataset#####
        #train_X, train_y, test_X, test_y = dataload(name)
        #train_XX, train_yy, test_XX, test_yy = dataload(name2)

        #####Take first value from test group (First value after training lot)#####
        #data.append(test_XX[0][0])
        #data = np.array(data)

        ######Generate 30 perturbations (states)#####
        perturb = self.perturbations(self.data, self.perturb)
        listado = []

        ######System evolution via Lorenz model (50 points each state)#####
        for i in range(30):
            todos = self.evolutionLorenz(perturb[i])
            listado.append(todos)

        listado = np.array(listado)

        ######Load model#####
        model = self.loadmodel(self.nombre)

        ######Initialization######
        H, sqR = self.initialization()

        for i in range(1000):
        #for i in range(2000):
            t51list = []
            y0 = []
            index = i
            ######Predict T+1 for each state#####
            for j in range(30):
                aux = listado[j]
            
                #reshape dimensions from (50, 3) to (1, 50, 3) (need for network prediction)
                aux = aux.reshape(-1, 50, 3)
            
                #make a prediction
                ypred = model.predict(aux)
                t51list.append(ypred)

            ######Transform list to array#####
            t51list = np.array(t51list)

            #####Prepare data for assimilation#####
            ######Vector of states (Xf)#####
            Xf = self.prepareData(t51list)
            ######True value t+1 (y0)#####
            y0 = self.test_yy[i+1]
            
            #Assimilation
            assT51 = self.perobs(Xf, y0, H, sqR)

            #Concatenate T51
            listado = self.concatenate(listado,assT51)
            
            #Save all data
            self.auxiliarDatos = self.todosDatos(listado,self.auxiliarDatos,index)

        #Split true values
        valXtrue, valYtrue, valZtrue = self.split_values(self.test_yy[951:1001])

        #Split obs values
        valXobs, valYobs, valZobs = self.split_values(self.test_y[951:1001])

        #RMSE
        promX, promY, promZ = self.promedio(listado)
        #rmse for each variable
        rmseX = sqrt(mean_squared_error(valXtrue, promX))
        rmseY = sqrt(mean_squared_error(valYtrue, promY))
        rmseZ = sqrt(mean_squared_error(valZtrue, promZ))

        #Plot vals
        name = 'Project/static/ass/ass.eps'
        name2 = 'Project/static/ass/ass.png'
        self.plotAll(valXtrue,valXobs,valYtrue,valYobs,valZtrue,valZobs)
        self.plot_results_ass(valXtrue,valYtrue,valZtrue,listado,rmseX,rmseY,rmseZ,name,name2)

        #Split true values
        valXtrue1000, valYtrue1000, valZtrue1000 = self.split_values(self.test_yy[1:1001])
        #Split obs values
        valXobs1000, valYobs1000, valZobs1000 = self.split_values(self.test_y[1:1001])
        #plot
        name = 'Project/static/ass/ass1000.eps'
        name2 = 'Project/static/ass/ass1000.png'
        self.plot_results_ass(valXtrue1000,valYtrue1000,valZtrue1000,self.auxiliarDatos,rmseX,rmseY,rmseZ,name,name2)

        return(valXtrue1000.tolist(),valYtrue1000.tolist(),valZtrue1000.tolist())