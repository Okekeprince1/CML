from django.urls import path
from . import views

urlpatterns = [
    path('', views.model_inference, name='model_inference'),
    path('compare/', views.model_comparison, name='model_comparison'),
] 