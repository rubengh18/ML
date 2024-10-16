from django.urls import path
from . import views

urlpatterns = [
    path('procesar_imagen/', views.procesar_imagen, name='procesar_imagen'),
    path("", views.index, name="index")
]
