from django import forms

class FormLorenz96II(forms.Form):
	forzado = forms.IntegerField(
		label="Constante forzada (F)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
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
			'placeholder': 'Valor'
			})
		)
	desechar = forms.IntegerField(
		label="Observaciones a desechar", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
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