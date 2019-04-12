from django import forms

class FormLorenz63(forms.Form):
	sigma = forms.IntegerField(
		initial="10",
		label="Valor de Sigma (Por defecto: 10)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	rho = forms.IntegerField(
		initial="28",
		label="Valor de Rho (Por defecto: 28)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	beta = forms.FloatField(
		initial="2.667",
		label="Valor de Beta (Por defecto: 2.667)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
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