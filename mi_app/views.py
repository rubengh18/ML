from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from .inimodel import model  # Asegúrate de que la importación del modelo esté correcta
from .utils import cargar_y_redimensionar_imagen, make_output
from django.shortcuts import render
import base64
import io
from PIL import Image
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    return render(request, "index.html")

@api_view(['POST'])
def procesar_imagen(request):
    if request.method == 'POST' and request.FILES.get('imagen'):
        imagen = request.FILES['imagen']
        ruta_imagen = os.path.join(BASE_DIR, 'temp', imagen.name)  # Guardar en una carpeta temporal dentro del proyecto
        
        with open(ruta_imagen, 'wb+') as destination:
            for chunk in imagen.chunks():
                destination.write(chunk)
        
        imagen_tensor = cargar_y_redimensionar_imagen(ruta_imagen)
        
        # Hacer la predicción
        predicciones = model.predict(imagen_tensor)
        etiqueta = make_output(predicciones)
        print(etiqueta)
        return JsonResponse({'etiqueta': etiqueta})
    
    return JsonResponse({'error': 'Imagen no recibida'}, status=400)


