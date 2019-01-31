from django import forms

class FormUnivar(forms.Form):
	epocas = forms.IntegerField(
		label="Numero de epocas", 
		widget=forms.TextInput(attrs={
			'class':'form-control ',
			'placeholder': 'Valor'
			})
		)
	ventana = forms.IntegerField(
		label="Tamaño por lotes (MiniBatchsize):", 
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
