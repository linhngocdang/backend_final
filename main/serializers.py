from rest_framework import serializers
from .models import Employee, Doctor
class EmployeeSerializer(serializers.ModelSerializer):
    class Meta:
        model=Employee
        fields = ['id', 'full_name','age', 'gender', 'email', 'contact', 'address','birthday','featured_img','classified','updated']
        read_only_fields = ['classified','updated']
        
class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model=Doctor
        fields = ['id', 'full_name', 'email', 'password']