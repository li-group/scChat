# Create your models here.
from django.db import models

class FileUpload(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    print("shit model created", title, file)

    def __str__(self):
        return self.title
