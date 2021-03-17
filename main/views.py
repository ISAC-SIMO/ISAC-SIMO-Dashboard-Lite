from django.shortcuts import render
from django.http.response import JsonResponse

def home(request):
    if request.method == "GET":
      return render(request, 'home.html', {'version':'1'})
    else:
      return JsonResponse(status=200, data={'version':'1'})