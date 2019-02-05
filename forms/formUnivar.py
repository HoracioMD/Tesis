from django import forms

class FormUnivar(forms.Form):
	nombre = forms.CharField(
		label="Nombre del dataset",
		required=True, 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
			'id': 'name',
			'readonly': 'readonly'
			})
		)
	epocas = forms.IntegerField(
		initial="10",
		required=True,
		label="Numero de épocas", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
			'readonly': 'readonly'
			})
		)
	ventana = forms.IntegerField(
		initial="200",
		required=True,
		label="Tamaño por lotes (Batch-size):", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor',
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
		widget=forms.Select, 
		choices=ACTIVATION, 
		required=True, 
		label='Función de activación',
		disabled=True)

	OPTIMIZATION = (
	('adam', 'ADAM'),
    ('rmsprop', 'RMSprop'),
    ('sgd','Stochastic gradient descent'))
	optimizar = forms.ChoiceField(
		initial='adam',
		widget=forms.Select, 
		choices=OPTIMIZATION, 
		required=True, 
		label='Algoritmo de optimización',
		disabled=True)

	LOSS = (
	('mse', 'Mean Squared Error'),
    ('mae', 'Mean Absolute Error'),
    ('mape','Mean Absolute Percentage Error'))
	perdidas = forms.ChoiceField(
		initial='mse',
		widget=forms.Select, 
		choices=LOSS, 
		required=True, 
		label='Función de perdida',
		disabled=True)