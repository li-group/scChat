from django.urls import re_path

websocket_urlpatterns = []

try:
    from . import consumers
    if hasattr(consumers, 'ChatConsumer'):
        websocket_urlpatterns = [
            re_path(r'ws/chat/(?P<room_name>\w+)/$', consumers.ChatConsumer.as_asgi()),
        ]
except ImportError:
    pass  # Channels not available