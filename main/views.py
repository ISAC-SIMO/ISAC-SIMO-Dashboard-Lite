import base64
import os
from django.shortcuts import render
from django.http.response import JsonResponse
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import cv2
import json
import numpy as np
import requests
import tensorflow as tf
from PIL import Image

temp_path = 'static/temp/'

def home(request):
    if request.method == "GET":
      return render(request, 'home.html', {'version':'1'})
    else:
      return JsonResponse(status=200, data={'version':'1'})

def test(request, useConfig=False):
    try:
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

        config = []
        config_file = temp_path + 'config.json'

        # Use Config (Example if coming from Mobile APP)
        if useConfig:
          if not os.path.exists(config_file) or not os.path.isfile(config_file):
            return JsonResponse(status=404, data={'message':'Config file does not exist.'})
          
          with open(config_file) as f:
            config = json.load(f)
            models = config

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
          model = m
          if type(m) is str: # String for Web App & Dict for config based API approach
            model = json.loads(m)
          
          file_name = None
          _file = None

          if useConfig: # For Mobile Config based API (file will be loaded differently)
            file_name = model.get("file")
            if file_name:
              with open(file_name) as f:
                _file = f
          else: # For Web App
            file = files.get('file_'+str(model.get('id'))) or None
            # Open File to read
            if file:
              file_name = temp_path + file.__str__()
              if type(file) is InMemoryUploadedFile:
                  file = file.open()
              else:
                  file = open(file.temporary_file_path(), 'rb')

              # Write the file in new temp file...
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

          if not useConfig:
            # Append new Config
            config.append({
              "id": str(model.get("id")),
              "name": str(model.get("name")),
              "type": str(model.get("type")),
              "file": file_name,
              "labels": model.get("labels", []),
              "collection_id": str(model.get("collection_id", None)),
              "ibm_api_key": str(model.get("ibm_api_key", None))
            })

          if res and type(res) is dict:
            response[str(model.get("id"))] = res
            pipeline[str(model.get("name"))] = res
            
            if res.get("path"): # Path value replaces old image_path
              image_path = res.get("path")
            
            if res.get("score"):
              score = res.get("score")

            if res.get("result"):
              result = res.get("result")

            if res.get("die"):
              print('Dieing..')
              break;
          else:
            response[str(model.get("id"))] = res
            pipeline[str(model.get("name"))] = res

        # Finally send response
        _image.close()

        response['pipeline'] = pipeline

        if not useConfig:
          # Write Config to File
          with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        else:
          # Saving Mobile Response (To show in UI all pipeline results)
          with open(temp_path + 'external_response.json', 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=4)

        if useConfig:
          # Little Bit Different Respones for Mobile API
          return JsonResponse(status=200, data={"data": response, "score": score, "result": result})
        else:
          return JsonResponse(status=200, data=response)
    except Exception as e:
      print(e)
      return JsonResponse(status=500, data={'message': str(e).title() or 'Invalid Request'})

    # If invalid request method
    return JsonResponse(status=404, data={'message':'Invalid Request'})

@csrf_exempt
def testMobile(request):
  return test(request, useConfig=True)

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
        return {"id": model.get("id"), "name": model.get("name"), "type": "preprocessor", "data" : 'data:image/jpg;base64,'+baseuri, "path": temp_path+image_name, "status": True}
      else:
        return {"id": model.get("id"), "name": model.get("name"), "type": "preprocessor", "message" : "No File Provided for Preprocessor", "die": True, "status": False}

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
        return {"id": model.get("id"), "name": model.get("name"), "type": "postprocessor", "data" : result, "score" : result.get('score', 0), "result" : result.get('result', ""), "die" : result.get('break', False), "status": True}
      else:
        return {"id": model.get("id"), "name": model.get("name"), "type": "postprocessor", "message" : "No File Provided for Postprocessor", "die": True, "status": False}

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
              return {"id": model.get("id"), "name": model.get("name"), "type": "classifier", "message" : "Failed to run h5, hdf5, keras, Offline Model", "die": True, "status": False}
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
              return {"id": model.get("id"), "name": model.get("name"), "type": "classifier", "message" : "Failed to run .py Offline Model", "die": True, "status": False}
      else:
        return {"id": model.get("id"), "name": model.get("name"), "type": "classifier", "message" : "This Offline Model Format is not supported.", "die": True, "status": False}

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
      return {"id": model.get("id"), "name": model.get("name"), "type": "classifier", 'data':data, 'score':score, 'result':result_type, "die": False, "status": True}

    # Type = watsonobjectdetection
    elif model.get("type") == "watsonobjectdetection":
      collection_id = model.get("collection_id")
      ibm_api_key = model.get("ibm_api_key")
      post_data = {'collection_ids': collection_id, 'features':'objects', 'threshold':'0.15'} # 'threshold': '0.15 -1'
      auth_base = 'Basic '+str(base64.b64encode(bytes('apikey:'+ibm_api_key, 'utf-8')).decode('utf-8'))
      post_header = {'Accept':'application/json','Authorization':auth_base}
      
      _image = open(image_path, 'rb')
      post_files = {
        'images_file': _image,
      }
      response = requests.post('https://gateway.watsonplatform.net/visual-recognition/api/v4/analyze?version=2019-02-11', files=post_files, headers=post_header, data=post_data)
      status = response.status_code
      try:
          content = response.json()
          if(status == 200 or status == '200' or status == 201 or status == '201' and content):
            if "collections" in content['images'][0]['objects']:
              if(content['images'][0]['objects']['collections'][0]['objects']):
                sorted_by_score = sorted(content['images'][0]['objects']['collections'][0]['objects'], key=lambda k: k['score'], reverse=True)
                # print(sorted_by_score)
                if(sorted_by_score and sorted_by_score[0]):
                  return {"id": model.get("id"), "name": model.get("name"), "type": "watsonobjectdetection", 'data':sorted_by_score, 'score':sorted_by_score[0]['score'], 'result':sorted_by_score[0]['object'], "die": False, "status": True}

          print("Reason:",status,content)
          return {"id": model.get("id"), "name": model.get("name"), "type": "watsonobjectdetection", "message" : "Watson Object Detection did not detect anything.", "die": True, "status": False}

      # Watson is not valid JSON etc
      except ValueError as e:
          # IBM Response is BAD
          print(e)
          print('IBM Watson Object Detection Response was BAD - (e.g. image too large, response json was invalid etc.)')
          return {"id": model.get("id"), "name": model.get("name"), "type": "watsonobjectdetection", "message" : "Watson Object Detection Test Failed.", "die": True, "status": False}
      finally:
        _image.close()

    # Type = watsonclassifier
    elif model.get("type") == "watsonclassifier":
      collection_id = model.get("collection_id")
      ibm_api_key = model.get("ibm_api_key")
      post_data = {'collection_ids': collection_id}
      auth_base = 'Basic '+str(base64.b64encode(bytes('apikey:'+ibm_api_key, 'utf-8')).decode('utf-8'))
      post_header = {'Accept':'application/json','Authorization':auth_base}
      
      _image = open(image_path, 'rb')
      post_files = {
        'images_file': _image,
      }
      response = requests.post('https://gateway.watsonplatform.net/visual-recognition/api/v3/classify?version=2018-03-19', files=post_files, headers=post_header, data=post_data)
      status = response.status_code
      try:
          content = response.json()
          if(content['images'][0]['classifiers'][0]['classes']):
            sorted_by_score = sorted(content['images'][0]['classifiers'][0]['classes'], key=lambda k: k['score'], reverse=True)
            return {"id": model.get("id"), "name": model.get("name"), "type": "watsonclassifier", 'data':sorted_by_score, 'score':sorted_by_score[0]['score'], 'result':sorted_by_score[0]['class'], "die": False, "status": True}

          print("Reason:",status,content)
          return {"id": model.get("id"), "name": model.get("name"), "type": "watsonclassifier", "message" : "Watson Classifier did not found anything.", "die": True, "status": False}

      # Watson is not valid JSON etc
      except ValueError as e:
          # IBM Response is BAD
          print(e)
          print('IBM Watson Classifier Response was BAD - (e.g. image too large, response json was invalid etc.)')
          return {"id": model.get("id"), "name": model.get("name"), "type": "watsonclassifier", "message" : "Watson Classifier Test Failed.", "die": True, "status": False}
      finally:
        _image.close()

    else:
      return {"id": model.get("id"), "name": model.get("name"), "type": model.get("type"), "message" : "Invalid Model Type Received", "die": True, "status": False}
  except Exception as e:
    print('An Error Occurred')
    print(e)
    # print(e.with_traceback(e.__traceback__))
    return {"id": model.get("id"), "name": model.get("name"), "type": model.get("type"), "message" : "An error occurred. Make sure the models are valid. Also, check Django Log for possible error.", "die": True, "status": False}