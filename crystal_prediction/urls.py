
from django.urls import path
from . import views
from .views import crystalStructurePredictor

urlpatterns = [
    path('', crystalStructurePredictor, name='crystal_structure_predictor'),
]
