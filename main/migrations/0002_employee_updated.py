# Generated by Django 4.2.2 on 2023-06-15 19:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='employee',
            name='updated',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
