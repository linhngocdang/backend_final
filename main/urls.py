from django.urls import path, include
from rest_framework import routers
from .views import EmployeeViewSet, DoctorViewSet, doctor_login
router = routers.DefaultRouter()
router.register(r'employee', EmployeeViewSet)
router.register(r'doctor', DoctorViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('doctor-login/', doctor_login, name='doctor-login'),
    
]
