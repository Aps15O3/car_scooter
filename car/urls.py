from django.contrib import admin
from django.urls import path
from car import views

urlpatterns = [
    path('', views.home),
    path("car",views.index),
    path('scooter',views.scooter)
]
