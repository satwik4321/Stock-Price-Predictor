"""
URL configuration for stockproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views.
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home.
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls') test comment
"""
from django.urls import path
from myapp import views
from django.urls import path
from myapp import views
#from myapp.views import handle_stock_submission

urlpatterns = [
    path('collect_history/', views.collect_history, name='collect_history'),
    #path('stock-form/', handle_stock_submission, name='stock_form'),
    path('', views.home, name='home')
]