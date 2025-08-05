import json

try:
    from channels.generic.websocket import AsyncWebsocketConsumer
    from channels.db import database_sync_to_async
    from asgiref.sync import async_to_sync
    from channels.layers import get_channel_layer
    CHANNELS_AVAILABLE = True
except ImportError:
    CHANNELS_AVAILABLE = False
    # Define dummy classes/functions to prevent import errors
    AsyncWebsocketConsumer = object
    get_channel_layer = lambda: None
    async_to_sync = lambda x: lambda *args, **kwargs: None

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        message_type = text_data_json.get('type', 'chat_message')

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message
            }
        )

    # Receive message from room group
    async def chat_message(self, event):
        message = event['message']

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': message
        }))

    # Handle progress updates
    async def progress_update(self, event):
        await self.send(text_data=json.dumps({
            'type': 'progress',
            'message': event['message'],
            'progress': event.get('progress', 0),
            'stage': event.get('stage', '')
        }))


def send_progress_update(room_name, message, progress=None, stage=None):
    """
    Helper function to send progress updates from synchronous code
    """
    if not CHANNELS_AVAILABLE:
        # If channels not available, just print the progress
        print(f"Progress [{room_name}]: {message} - {progress}% ({stage})")
        return
        
    channel_layer = get_channel_layer()
    if channel_layer:
        async_to_sync(channel_layer.group_send)(
            f'chat_{room_name}',
            {
                'type': 'progress_update',
                'message': message,
                'progress': progress,
                'stage': stage
            }
        )