from django.shortcuts import render
from django.http import HttpResponse


def home(request):
    return render(request,'web_app/home.html')



# def about(request):
#     return render(request,'blog/about.html',{'title' : 'About'})




