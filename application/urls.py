from django.contrib import admin
from django.urls import path
from main import views

urlpatterns = [
    path('', views.home, name='home'),
    path('test/', views.test, name='test'),
    path('api/test/', views.testMobile, name='api.test'),
    path('admin/', admin.site.urls),
]
