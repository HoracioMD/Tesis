from django import forms

class FormLorenz96II(forms.Form):
	forzado = forms.IntegerField(
		label="Constante forzada (F)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Mas grande, mas caótico el sistema'
			})
		)
	x_big_scale = forms.IntegerField(
		initial="8",
		label="Cantidad de variables de X", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
			'readonly': 'readonly'
			})
		)
	y_small_scale = forms.IntegerField(
		initial="256",
		label="Cantidad de variables de Y", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
			'readonly': 'readonly',
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