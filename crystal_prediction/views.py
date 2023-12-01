from django.shortcuts import render
from .forms import CrystalStructureForm
from .predict import predict_crystal_structure  # You need to implement this function

def crystalStructurePredictor(request):
    if request.method == 'POST':
        form = CrystalStructureForm(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            prediction = predict_crystal_structure(instance)  # Implement this function
            instance.predicted_xgb = prediction  # Assuming you have a predicted_xgb field in your model
            instance.save()
    else:
        form = CrystalStructureForm()

    return render(request, 'crystal_prediction/crystal_structure_predictor.html', {'form': form , 'instance':instance})