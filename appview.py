from django.shortcuts import render, HttpResponse
from django.template import RequestContext
from django.template.defaulttags import csrf_token
import json
import netCDF4 as nc4
from .netcdf4 import NCDF4

from .modules.lorenz96.L96I import Lorenz96I
from .modules.lorenz96.L96II import Lorenz96II
from .modules.lorenz96.L63 import Lorenz63
from .modules.lorenz96.l63error import Lorenz63error

#from .rnn.univariable.predict_l96I import PredictL96I
from .rnn.univariable.Auxpredict_l96I import PredictL96I
from .rnn.multivariable.predict_l63 import PredictL63
from .rnn.multivariable.parameters63 import ParametersL63
from .rnn.univariable.parameters96 import ParametersL96
from .rnn.assimilation.prediction import prediction
from .rnn.assimilation.assimilation import assimilation

from .forms.formL96I import FormLorenz96I
from .forms.formL96II import FormLorenz96II
from .forms.formL63 import FormLorenz63
from .forms.formUnivar import FormUnivar
from .forms.formMultivar import FormMultivar
from .forms.formParam import FormParameters
from .forms.formL63error import FormLorenz63Error


def assPred(request):
	send = True
	if request.method == "POST":
		predAss = prediction()
		array = predAss.startPred()
		return HttpResponse(
		json.dumps(array),
        content_type="application/json")

def assimilate(request):
	send = True
	if request.method == "POST":
		asimilated = assimilation()
		array = asimilated.main()
		return HttpResponse(
		json.dumps(array),
        content_type="application/json")

def l63error(request):
	send = False
	if request.method == "POST":
		form = FormLorenz63Error(request.POST)
		if form.is_valid():
			send = True
			sigma = form.cleaned_data["sigma"]
			rho = form.cleaned_data["rho"]
			beta = form.cleaned_data["beta"]
			errorXY = form.cleaned_data["errorXY"]
			errorZ = form.cleaned_data["errorZ"]
			obs = form.cleaned_data["observaciones"]

			sigma, rho, beta, errorXY, errorZ, obs
			object_l63e = Lorenz63error(sigma, rho, beta, errorXY, errorZ, obs)
			array = object_l63e.l63error()
			#[arrayX, arrayY] = object_l96I.l96I(desechar)
			return HttpResponse(
			json.dumps(array),
            content_type="application/json")

def parameters(request):
	send = False
	if request.method == "POST":
		form = FormParameters(request.POST)
		if form.is_valid():
			send = True
			epocas = form.cleaned_data["epocas"]
			ventana = form.cleaned_data["ventana"]
			dropout = form.cleaned_data["dropout"]
			lrate = form.cleaned_data["lrate"]
			activar = form.cleaned_data["activar"]
			optimizar = form.cleaned_data["optimizar"]
			perdidas = form.cleaned_data["perdidas"]
			#guardar = form.cleaned_data["guardar"]

			epocas, ventana, dropout, lrate, activar, optimizar, perdidas
			objeto = ParametersL63(epocas, ventana, dropout, lrate, activar, optimizar, perdidas)
			objeto2 = ParametersL96(epocas, ventana, dropout, lrate, activar, optimizar, perdidas)
			return HttpResponse(
			json.dumps(111),
            content_type="application/json")

def l96II(request):
	send = False
	if request.method == "POST":
		form = FormLorenz96II(request.POST)
		if form.is_valid():
			send = True
			F = form.cleaned_data["forzado"]
			X = form.cleaned_data["x_big_scale"]
			Y = form.cleaned_data["y_small_scale"]
			obs = form.cleaned_data["observaciones"]
			desechar = form.cleaned_data["desechar"]
			guardar = form.cleaned_data["guardar"]

			X, Y, F, obs, guardar
			object_l96II = Lorenz96II(X, Y, F, obs, guardar)
			array = object_l96II.l96II(desechar)
			
			return HttpResponse(
			json.dumps(array),
            content_type="application/json")

def l96I(request):
	send = False
	if request.method == "POST":
		form = FormLorenz96I(request.POST)
		if form.is_valid():
			send = True
			F = form.cleaned_data["forzado"]
			N = form.cleaned_data["x_small_scale"]
			obs = form.cleaned_data["observaciones"]
			desechar = form.cleaned_data["desechar"]
			guardar = form.cleaned_data["guardar"]

			N, F, obs, guardar
			object_l96I = Lorenz96I(N, F, obs, guardar)
			array = object_l96I.l96I(desechar)
			#[arrayX, arrayY] = object_l96I.l96I(desechar)
			return HttpResponse(
			json.dumps(array),
            content_type="application/json")

def l63(request):
	send = False
	if request.method == "POST":
		form = FormLorenz63(request.POST)
		if form.is_valid():
			send = True
			sigma = form.cleaned_data["sigma"]
			rho = form.cleaned_data["rho"]
			beta = form.cleaned_data["beta"]
			obs = form.cleaned_data["observaciones"]
			desechar = form.cleaned_data["desechar"]
			guardar = form.cleaned_data["guardar"]

			sigma, rho, beta, obs, guardar
			object_l63 = Lorenz63(sigma, rho, beta, obs, guardar)
			array = object_l63.l63(desechar)
			#[arrayX, arrayY] = object_l96I.l96I(desechar)
			return HttpResponse(
			json.dumps(array),
            content_type="application/json")

def l96predict(request):
	send = False
	if request.method == "POST":
		form = FormUnivar(request.POST)
		#print(form)
		if form.is_valid():
			send = True
			nombre = form.cleaned_data["nombre"]
			epocas = form.cleaned_data["epocas"]
			ventana = form.cleaned_data["ventana"]
			dropout = form.cleaned_data["dropout"]
			lrate = form.cleaned_data["lrate"]
			activar = form.cleaned_data["activar"]
			optimizar = form.cleaned_data["optimizar"]
			perdidas = form.cleaned_data["perdidas"]
			#guardar = form.cleaned_data["guardar"]

			nombre, epocas, ventana, dropout, lrate, activar, optimizar, perdidas
			object_predictl96I = PredictL96I(nombre, epocas, ventana, dropout, lrate, activar, optimizar, perdidas)
			#object_predictl96I = PredictL96I(nombre)
			array = object_predictl96I.start_prediction()
			#[arrayX, arrayY] = object_l96I.l96I(desechar)
			return HttpResponse(
			json.dumps(array),
            content_type="application/json")

def l63predict(request):
	send = False
	if request.method == "POST":
		form = FormUnivar(request.POST)
		#print(form)
		if form.is_valid():
			send = True
			nombre = form.cleaned_data["nombre"]
			epocas = form.cleaned_data["epocas"]
			ventana = form.cleaned_data["ventana"]
			dropout = form.cleaned_data["dropout"]
			lrate = form.cleaned_data["lrate"]
			activar = form.cleaned_data["activar"]
			optimizar = form.cleaned_data["optimizar"]
			perdidas = form.cleaned_data["perdidas"]
			#guardar = form.cleaned_data["guardar"]

			nombre, epocas, ventana, dropout, lrate, activar, optimizar, perdidas
			object_predictl96I = PredictL63(nombre, epocas, ventana, dropout, lrate, activar, optimizar, perdidas)
			#object_predictl96I = PredictL63(nombre)
			array = object_predictl96I.start_prediction()
			#[arrayX, arrayY] = object_l96I.l96I(desechar)
			return HttpResponse(
			json.dumps(array),
            content_type="application/json")

def graph_l96I_dataset(request):
	filename = request.POST.get('filename')
	nc = NCDF4()
	graficar = nc.extract_l96I_data(filename)
	return HttpResponse(
	json.dumps(graficar), 
	content_type="application/json")

def graph_l63_dataset(request):
	filename = request.POST.get('filename')
	nc = NCDF4()
	graficar = nc.extract_l63_data(filename)
	return HttpResponse(
	json.dumps(graficar), 
	content_type="application/json")

def graph_l96II_dataset(request):
	filename = request.POST.get('filename')
	nc = NCDF4()
	graficar = nc.extract_l96II_data(filename)
	return HttpResponse(
	json.dumps(graficar), 
	content_type="application/json")

def graph_l96Ipred_dataset(request):
	filename = request.POST.get('filename')
	nc = NCDF4()
	graficar = nc.extract_l96Ipred_data(filename)
	return HttpResponse(
	json.dumps(graficar), 
	content_type="application/json")

def graph_l63pred_dataset(request):
	filename = request.POST.get('filename')
	nc = NCDF4()
	graficar = nc.extract_l63pred_data(filename)
	return HttpResponse(
	json.dumps(graficar), 
	content_type="application/json")