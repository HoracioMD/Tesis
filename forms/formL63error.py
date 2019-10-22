from django import forms

class FormLorenz63Error(forms.Form):
	sigma = forms.IntegerField(
		initial="10",
		required=True,
		label="Valor de Sigma (Por defecto: 10)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	rho = forms.IntegerField(
		initial="28",
		required=True,
		label="Valor de Rho (Por defecto: 28)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	beta = forms.FloatField(
		initial="2.667",
		required=True,
		label="Valor de Beta (Por defecto: 2.667)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	errorXY = forms.IntegerField(
		initial="3",
		required=True,
		label="Magnitud del error para X e Y", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	errorZ = forms.IntegerField(
		initial="6",
		required=True,
		label="Magnitud del error para Z", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	observaciones = forms.IntegerField(
		required=True,
		label="Total de observaciones a generar", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Minimo 100000 valores'
			})
		)