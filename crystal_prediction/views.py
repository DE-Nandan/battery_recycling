from django.shortcuts import render
from django.http import JsonResponse
from .forms import CrystalStructureForm
from .predict import predict_crystal_structure  # You need to implement this function

def crystalStructurePredictor(request):
    if request.method == 'POST':
        form = CrystalStructureForm(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            prediction_method = form.cleaned_data.get('prediction_method')
            
            if(prediction_method == 'neural_network'):
                 return JsonResponse({'processing': True})

            prediction = predict_crystal_structure(instance,prediction_method)  
            instance.predicted_xgb = prediction
            instance.save()
            return render(request, 'crystal_prediction/crystal_structure_predictor.html', {'form': form , 'instance':instance })

    else:
        form = CrystalStructureForm()

    return render(request, 'crystal_prediction/crystal_structure_predictor.html', {'form': form })