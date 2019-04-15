from django import forms

class FormLorenz96I(forms.Form):
	forzado = forms.IntegerField(
		label="Constante forzada (F)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Mas grande, mas caótico el sistema'
			})
		)
	x_small_scale = forms.IntegerField(
		label="Cantidad de variables (N)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor aconsejable: 12'
			})
		)
	observaciones = forms.IntegerField(
		label="Observaciones a dibujar", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Cantidad de valores a plotear'
			})
		)
	desechar = forms.IntegerField(
		label="Total de observaciones a generar", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Minimo 2000 valores'
			})
		)
	PART_CHOICES = (
    ('1', 'Si'),
    ('2', 'No'),)
	guardar = forms.ChoiceField(
		widget=forms.RadioSelect, 
		choices=PART_CHOICES, 
		required=True, 
		label='Guardar Datos')
