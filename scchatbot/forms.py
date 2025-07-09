# forms.py
from django import forms
from scchatbot.models import FileUpload

class MyForm(forms.Form):
    my_field = forms.CharField(label='Enter something', max_length=100)

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = FileUpload
        fields = ['title', 'file']
