#from django.template.loader import get_template
#from django.template import Context
#from django.http import HttpResponse
from django.shortcuts import render_to_response, render

from .forms.formL96I import FormLorenz96I
from .forms.formL96II import FormLorenz96II
from .forms.formL63 import FormLorenz63
from .forms.formUnivar import FormUnivar
from .netcdf4 import NCDF4

def principal(request):
	return render_to_response('indx.html')

def tablas(request):
	objeto = NCDF4()
	return render(request, 'tablas.html', {'list_l96I': objeto.return_list_nc('l96I')})

def tablas63(request):
	objeto = NCDF4()
	return render(request, 'tablasL63.html', {'list_l63': objeto.return_list63_nc('l63')})

def tablas2(request):
	objeto = NCDF4()
	return render(request, 'tablasL96II.html', {'list_l96II': objeto.return_list2_nc('l96II')})

def analisisPredL96I(request):
	objeto = NCDF4()
	return render(request, 'l96Ianalisis.html', {'list_l96Ipred': objeto.return_list_pred_nc('l96Ipred')})

def lorenz(request):
	return render(request,'lorenz.html', {'form': FormLorenz96I()})

def lorenz2(request):
	return render(request,'lorenz2.html', {'form': FormLorenz96II()})

def lorenz63(request):
	return render(request,'lorenz63.html', {'form': FormLorenz63()})

def hiperparam(request):
	objeto = NCDF4()
	return render(request,'parameter.html', {'list_l96I': objeto.return_list_nc('l96I'), 'form': FormUnivar()})

def univarLSTM(request):
	objeto = NCDF4()
	return render(request, 'univarLSTM.html', {'list_l96I': objeto.return_list_nc('l96I'), 'form': FormUnivar()})
