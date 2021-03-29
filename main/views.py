import base64
import os
from django.shortcuts import render
from django.http.response import JsonResponse
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.conf import settings
import cv2
import json
import numpy as np
import tensorflow as tf
from PIL import Image

temp_path = 'static/temp/'

def home(request):
    if request.method == "GET":
      return render(request, 'home.html', {'version':'1'})
    else:
      return JsonResponse(status=200, data={'version':'1'})

def test(request):
    if request.method == "POST":
      # Variables:
      # image, image_name, _image (open file), image_path (saved path)
      # file, file_name, _file (saved file) -- every for each models
      # models = json string coming from frontend
      # response = dict of response to send
      # score = Score value (0.5, 1, 0.2)
      # result = Result for the score (go,nogo)
      # pipeline = Similar to response for with name etc.

      # print(request.POST)
      models = request.POST.getlist('models[]')
      files = request.FILES
      image = request.FILES.get("image")
      _image = None

      if not image:
        return JsonResponse(status=404, data={'message':'Image Not Provided'})
      image_name = image.__str__()

      img = Image.open(image)
      img = img.convert('RGB')
      img.save(temp_path+image_name, format='JPEG', quality=100)
      image_path = temp_path+image_name

      # Open Image to read
      if type(image) is InMemoryUploadedFile:
          _image = image.open()
      else:
          _image = open(image.temporary_file_path(), 'rb')

      response = {}
      score = 0
      result = ""
      pipeline = {}

      # print(files)

      for m in models:
        model = json.loads(m)
        file = files.get('file_'+str(model.get('id'))) or None
        file_name = None
        # print(model)
        # print(file)

        # Open File to read
        if file:
          file_name = temp_path + file.__str__()
          if type(file) is InMemoryUploadedFile:
              file = file.open()
          else:
              file = open(file.temporary_file_path(), 'rb')

          # Write the file in new temp file...
          _file = None
          flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC |
                  getattr(os, 'O_BINARY', 0))
          # The current umask value is masked out by os.open!
          fd = os.open(file_name, flags, 0o666)
          try:
              for chunk in file.chunks():
                  if _file is None:
                      mode = 'wb' if isinstance(chunk, bytes) else 'wt'
                      _file = os.fdopen(fd, mode)
                  _file.write(chunk)
          finally:
              if _file is not None:
                  _file.close()
              else:
                  os.close(fd)
              file.close()
          
          # print(_file)
              
        ############
        # RUN TEST #
        ############
        res = run_test(model, _file, file_name, image, image_name, image_path, score, result, pipeline)

        if res and type(res) == "dict":
          response[str(model.get("id"))] = res
          pipeline[str(model.get("name"))] = res
          
          if res.get("path"): # Path value replaces old image_path
            image_path = res.get("path")

          if res.get("die"):
            print('Dieing..')
            return JsonResponse(status=200, data=response)
        else:
          response[str(model.get("id"))] = res
          pipeline[str(model.get("name"))] = res

      # Finally send response
      _image.close()
      response['pipeline'] = pipeline
      return JsonResponse(status=200, data=response)
    
    # If invalid request method
    return JsonResponse(status=404, data={'message':'Invalid Request'})

def run_test(model, file, file_name, image, image_name, image_path, score, result, pipeline):
  try:
    # Type = preprocessor
    if model.get("type") == "preprocessor":
      if(file and file_name):
        # print(file_name)
        import importlib
        loader = importlib.machinery.SourceFileLoader('model', file_name)
        handle = loader.load_module('model')

        # If Image Path exists i.e. already saved specially by preprocessor
        if image_path:
          img = cv2.imread(image_path)
        else:
          img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        result = handle.run(img)
        baseuri = base64.b64encode(cv2.imencode('.jpg', result)[1]).decode()
        cv2.imwrite(temp_path+image_name, result)
        return {"data" : 'data:image/jpg;base64,'+baseuri, "path": temp_path+image_name, "status": True}
      else:
        return {"message" : "No File Provided for Preprocessor", "die": True, "status": False}

    # Type = postprocessor
    elif model.get("type") == "postprocessor":
      if(file and file_name):
        # print(file_name)
        import importlib
        loader = importlib.machinery.SourceFileLoader('model', file_name)
        handle = loader.load_module('model')

        # If Image Path exists i.e. already saved specially by preprocessor
        if image_path:
          img = cv2.imread(image_path)
        else:
          img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        result = handle.run(img,pipeline,score,result)
        return {"data" : result, "die" : result.get('break', False), "status": True}
      else:
        return {"message" : "No File Provided for Postprocessor", "die": True, "status": False}

    # Type = classifier
    elif model.get("type") == "classifier":
      labels = model.get("labels")
      img = Image.open(image_path).resize((150, 150))
      x = np.array(img)/255.0
      result = [[]]

      if ('h5', 'hdf5', 'keras').__contains__(file_name.split(".")[-1]): # tf.keras models
          try:
              new_model = tf.keras.models.load_model(file_name)
              result = new_model.predict(x[np.newaxis, ...])
              if type(result) != list:
                  result = result.tolist()
          except Exception as e:
              img.close()
              print(e)
              return {"message" : "Failed to run h5, hdf5, keras, Offline Model", "die": True, "status": False}
      elif ('py').__contains__(file_name.split(".")[-1]): # python script
          # Trying to run saved offline .py model (loading saved py file using string name)
          try:
              import importlib
              loader = importlib.machinery.SourceFileLoader('model', file_name)
              handle = loader.load_module('model')
              result = handle.run(img, labels)
              if type(result) is not list or len(result) <= 0:
                  result = [[]]
                  print('py local model -- BAD response (not list)')
          except Exception as e:
              print(e)
              return {"message" : "Failed to run .py Offline Model", "die": True, "status": False}
      else:
        return {"message" : "This Offline Model Format is not supported.", "die": True, "status": False}

      data = []
      
      i = 0
      for r in result[0]:
          if len(labels) > i:
              label = labels[i]
          else:
              label = 'No.'+str(i+1)

          data.append({
              "class": label,
              "score": r,
              "location": result[1][i] if (len(result) > 1 and len(result[1]) > i and type(result[1][i]) == dict) else None
          })
          i += 1

      result_type = ''
      score = '0'
      if len(result[0]) > 0:
          try:
              result_type = labels[result[0].index(max(result[0]))].lower()
          except:
              result_type = 'no.'+str(result[0].index(max(result[0])) + 1)
          finally:
              score = max(result[0])
      
      tf.keras.backend.clear_session()
      # tf.reset_default_graph()
      img.close()
      return {'data':data, 'score':score, 'result':result_type, "die": False, "status": True}

    else:
      return {"message" : "Invalid Model Type Received", "die": True, "status": False}
  except Exception as e:
    print('An Error Occurred')
    print(e)
    # print(e.with_traceback(e.__traceback__))
    return {"message" : "An error occurred. Check Django log.", "die": True, "status": False}