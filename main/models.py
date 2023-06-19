from django.db import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import os
import cv2

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

    # @staticmethod
    # def crop_image_from_gray(img, tol=7):
    #     if img.ndim == 2:
    #         mask = img > tol
    #         return img[np.ix_(mask.any(1), mask.any(0))]
    #     elif img.ndim == 3:
    #         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #         mask = gray_img > tol
            
    #         check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    #         if check_shape == 0: 
    #             return img 
    #         else:
    #             img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    #             img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    #             img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    #             img = np.stack([img1, img2, img3], axis=-1)
    #         return img

    def save(self, *args, **kwargs):
        try:
            model_file = 'hrd_model.h5'
            model_path = os.path.join(os.getcwd(), model_file)
            print('Saving employee:', self.full_name)
            print('Image path:', self.featured_img.path)
            if os.path.isfile(model_path):
                print("Model file exists.")
            else:
                print("Model file does not exist.")
            img_path = self.featured_img.path
            #img = load_img(img_path)
            # img_array = self.crop_image_from_gray(img, tol=7)
            img = load_img(img_path, target_size=(512, 512))
            img_array = img_to_array(img)
            #img_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_array = img_array / 255.0 
            img_array = np.expand_dims(img_array, axis=0)
            model = tf.keras.models.load_model('hrd_model.h5')
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