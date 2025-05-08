from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_with_ai, name='chat_with_ai'),
    path('upload/', views.upload_file, name='upload_file'),
    path('umap_leiden/', views.umap_leiden_view, name='umap_leiden'),
]