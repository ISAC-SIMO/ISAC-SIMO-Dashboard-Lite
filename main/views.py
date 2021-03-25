from django.shortcuts import render
from django.http.response import JsonResponse
import json

def home(request):
    if request.method == "GET":
      return render(request, 'home.html', {'version':'1'})
    else:
      return JsonResponse(status=200, data={'version':'1'})

def test(request):
    if request.method == "POST":
      # print(request.POST)
      models = request.POST.getlist('models[]')
      files = request.FILES
      image = request.FILES.get("image")

      if not image:
        return JsonResponse(status=404, data={'message':'Image Not Provided'})

      response = {}

      # print(files)

      for m in models:
        model = json.loads(m)
        file = files.get('file_'+str(model.get('id'))) or None
        print(model)
        print(file)
        res = run_test(model, file)

        if res and type(res) == "dict":
          response[str(model.get("id"))] = res
          if res.get("die"):
            return JsonResponse(status=200, data={'data': response})
        else:
          response[str(model.get("id"))] = res

      # Finally send response
      return JsonResponse(status=200, data={'data': response})
    
    # If invalid request method
    return JsonResponse(status=404, data={'message':'Invalid Request'})

def run_test(model, file):
  # Type = preprocessor
  if model.get("type") == "preprocessor":
    if(file):
      return True
    else:
      return {"message" : "No File Provided for Preprocessor", "die": True}

  return True