from django import forms

class FormParameters(forms.Form):
	epocas = forms.IntegerField(
		initial="15",
		required=True,
		label="Numero de épocas", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
			'id': 'epoch',
			'readonly': 'readonly'
			})
		)
	ventana = forms.IntegerField(
		initial="256",
		required=True,
		label="Tamaño por lotes (Batch-size):", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
			'id': 'window',
			'readonly': 'readonly'
			})
		)
	dropout = forms.FloatField(
		initial="0.2",
		required=True,
		label="Dropout (evitar sobreajuste)", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
			'id': 'drop',
			'readonly': 'readonly'
			})
		)
	lrate = forms.FloatField(
		initial="0.001",
		required=True,
		label="Taza de aprendizaje", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
			'id':'rate',
			'readonly': 'readonly'
			})
		)
	ACTIVATION = (
    ('linear', 'Lineal'),
    ('exponential', 'Exponencial'),
    ('sigmoid','Sigmoidea'),
    ('tanh', 'Tang. Hiperb.'))
	activar = forms.ChoiceField(
		initial='linear',
		#widget=forms.Select
		#(attrs={'id':'activation', 'class':'form-control '}), 
		widget=forms.RadioSelect(attrs={'id':'activation'}),
		choices=ACTIVATION, 
		required=True, 
		label='Función de activación',
		disabled=False)

	OPTIMIZATION = (
	('adam', 'ADAM'),
    ('rmsprop', 'RMSprop'),
    ('sgd','Stochastic gradient descent'))
	optimizar = forms.ChoiceField(
		initial='adam',
		#widget=forms.Select
		#(attrs={'id':'optimization', 'class':'form-control '}),
		widget=forms.RadioSelect(attrs={'id':'optimization'}), 
		choices=OPTIMIZATION, 
		required=True, 
		label='Algoritmo de optimización',
		disabled=False)

	LOSS = (
	('mse', 'Mean Squared Error'),
    ('mae', 'Mean Absolute Error'),
    ('mape','Mean Absolute Percentage Error'))
	perdidas = forms.ChoiceField(
		initial='mse',
		#widget=forms.Select
		#(attrs={'id':'loss', 'class':'form-control '}), 
		widget=forms.RadioSelect(attrs={'id':'loss'}),
		choices=LOSS, 
		required=True, 
		label='Función de perdida',
		disabled=False)