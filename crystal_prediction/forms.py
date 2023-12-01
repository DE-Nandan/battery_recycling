from django import forms
from .models import CrystalStructure

class CrystalStructureForm(forms.ModelForm):
    class Meta:
        model = CrystalStructure