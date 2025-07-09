from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_with_ai, name='chat_with_ai'),
    path('upload/', views.upload_file, name='upload_file'),
    path('get_umap_plot/', views.get_umap_plot, name='get_umap_plot'),
]