from django.db import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import model_from_json
import numpy as np
import tensorflow as tf
import os
import json

class Employee(models.Model):
    full_name = models.CharField(max_length=200)
    age = models.IntegerField(null=True)
    gender = models.CharField(max_length=20, null=True, blank=True)
    email = models.CharField(max_length=200, null=True, blank=True)
    contact = models.CharField(max_length=20)
    address = models.TextField()
    birthday = models.CharField(max_length=20, null=True, blank=True)
    featured_img = models.ImageField()
    classified = models.CharField(max_length=200, blank=True)
    updated = models.DateTimeField(auto_now=True, auto_now_add=False)

    def __str__(self):
        return self.full_name

    def save(self, *args, **kwargs):
        try:
            json_file = 'modelResnet.json'
            with open(json_file, 'r') as file:
                model_json = file.read()


            model = model_from_json(model_json)

            model.load_weights(json_file)

            img_path = self.featured_img.path
            img = load_img(img_path, target_size=(512, 512))
            img_array = img_to_array(img)
            img_array = img_array / 255.0 
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            class_id = np.argmax(prediction)
            class_names = ['NO DR', 'MILD', 'MODERATE', 'SEVERE', 'PROLIFERATIVE DR']
            class_name = class_names[class_id]
            self.classified = class_name
            print('Classification success')
        except Exception as e:
            print('Classification failed:', e)
        super().save(*args, **kwargs)

class Doctor(models.Model):
    full_name = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    password = models.CharField(max_length=100)
