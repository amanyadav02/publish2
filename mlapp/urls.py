from django.contrib import admin
from django.urls import path,include
from mlapp import views

urlpatterns = [
     path('',views.services,name='services'),
     path('home/',views.home,name='home'),

]