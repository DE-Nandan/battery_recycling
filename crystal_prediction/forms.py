from django import forms
from .models import CrystalStructure

class CrystalStructureForm(forms.ModelForm):

    PREDICTION_CHOICES = [
        ('xgb', 'XGBoost'),
        ('decision_tree', 'Decision Tree'),
        ('neural_network', 'Neural Network'),
        ('random_forest', 'Random Forest'),
    ]

    prediction_method = forms.ChoiceField(
        choices=PREDICTION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label='Prediction Method'
    )

    class Meta:
        model = CrystalStructure
        fields = '__all__'