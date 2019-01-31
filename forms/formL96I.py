from django import forms

class FormLorenz96I(forms.Form):
	forzado = forms.IntegerField(
		label="Constante forzada (F)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	x_small_scale = forms.IntegerField(
		label="Cantidad de variables (N)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
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
