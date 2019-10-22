"""Project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from Project.views import principal, lorenz, lorenz2, lorenz63, univarLSTM, multivarLSTM, multivar2LSTM, tablas, tablas63, tablas2, analisisPredL96I, analisisPredL63, hiperparam, hiperparam63, manual, perfil, logout, hiperparameters, assimilation
import Project.appview as app

urlpatterns = [
    url(r'^principal/$', principal, name='Principal'),
    url(r'^navegacion/$', manual, name='Manual'),
    url(r'^perfil/$', perfil, name='Perfil'),
    url(r'^parameters/$', hiperparameters, name='Parametros'),
    url(r'^logout/$', logout, name='Logout'),
    url(r'^tablas/$', tablas, name='Tablas'),
    url(r'^tablas63/$', tablas63, name='Tablas63'),
    url(r'^tablas2/$', tablas2, name='Tablas2'),
    url(r'^lorenz/$', lorenz, name='Lorenz96'),
    url(r'^lorenz2/$', lorenz2, name='Lorenz96II'),
    url(r'^lorenz63/$', lorenz63, name='Lorenz63'),        
    url(r'^predictUnivar/$', univarLSTM, name='UnivarLSTM'),
    url(r'^predictMultivar/$', multivarLSTM, name='MultivarLSTM'),
    url(r'^predictMultivar2/$', multivar2LSTM, name='Multivar2LSTM'),
    url(r'^analisisL96I/$', analisisPredL96I, name='AnalisisPredL96I'),
    url(r'^analisisL63/$', analisisPredL63, name='AnalisisPredL63'),
    url(r'^assimilation/$', assimilation, name='Asimilacion'),
    #url(r'^hiperparameters/$', hiperparam, name='Hiperparametros'),
    #url(r'^hiperparameters63/$', hiperparam63, name='Hiperparametros63'),

    url(r'^form/$', app.l96I, name='lorenz1'),
    url(r'^form2/$', app.l96II, name='lorenz2'),
    url(r'^form3/$', app.l63, name='lorenz3'),
    url(r'^form4/$', app.l96predict, name='lorenz96pred'),
    url(r'^form5/$', app.l63predict, name='lorenz63pred'),
    url(r'^form6/$', app.graph_l96I_dataset, name='graph_l96I_dataset'),
    url(r'^form7/$', app.graph_l63_dataset, name='graph_l63_dataset'),
    url(r'^form8/$', app.graph_l96II_dataset, name='graph_l96II_dataset'),
    url(r'^form9/$', app.graph_l96Ipred_dataset, name='graph_l96Ipred_dataset'),
    url(r'^form10/$', app.graph_l63pred_dataset, name='graph_l63pred_dataset'),
    url(r'^form11/$', app.parameters, name='saveparameters'),
    url(r'^form12/$', app.l63error, name='lorenz63error'),
    url(r'^form13/$', app.assPred, name='assimilationPred'),
    url(r'^form14/$', app.assimilate, name='assimilation'),

]
