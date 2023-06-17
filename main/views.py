from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets
from rest_framework.response import Response
from .models import Employee, Doctor
from .serializers import EmployeeSerializer, DoctorSerializer


class EmployeeViewSet(viewsets.ModelViewSet):
    queryset = Employee.objects.all().order_by('id')
    serializer_class = EmployeeSerializer

class DoctorViewSet(viewsets.ModelViewSet):
    queryset = Doctor.objects.all().order_by('id')
    serializer_class = DoctorSerializer
    

@csrf_exempt
def doctor_login(request):
    email=request.POST['email']
    password=request.POST['password']
    try:
        doctorData = Doctor.objects.get(email=email, password=password)
    except Doctor.DoesNotExist:
        doctorData=None
    if doctorData:
        return JsonResponse({'bool':True, 'doctor_id': doctorData.id})
    else:
        return JsonResponse({'bool':False})