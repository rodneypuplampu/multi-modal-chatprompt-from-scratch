# views.py
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from google.cloud import storage, dlp_v2
import json

@login_required
def chat_room(request, room_id):
    room = ChatRoom.objects.get(id=room_id)
    messages = Message.objects.filter(room=room).order_by('created_at')
    return render(request, 'chat/room.html', {
        'room': room,
        'messages': messages
    })

@login_required
def upload_message(request):
    if request.method == 'POST':
        room_id = request.POST.get('room_id')
        content = request.POST.get('content')
        media_file = request.FILES.get('media_file')
        media_type = request.POST.get('media_type')
        
        # Create message
        message = Message.objects.create(
            room_id=room_id,
            user=request.user,
            content=content,
            media_type=media_type
        )
        
        if media_file:
            # Upload to Google Cloud Storage
            message.media_file.save(
                f"{message.id}_{media_file.name}",
                media_file
            )
        
        # Perform DLP analysis on text content
        if content and media_type == 'text':
            dlp_client = dlp_v2.DlpServiceClient()
            parent = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}"
            
            item = {"value": content}
            
            # Configure what content you want to inspect
            inspect_config = {
                "info_types": [
                    {"name": "PHONE_NUMBER"},
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "CREDIT_CARD_NUMBER"},
                ]
            }
            
            response = dlp_client.inspect_content(
                request={
                    "parent": parent,
                    "inspect_config": inspect_config,
                    "item": item,
                }
            )
            
            message.dlp_analyzed = True
            message.save()
            
        return JsonResponse({
            'status': 'success',
            'message_id': message.id
        })
    
    return JsonResponse({'status': 'error'}, status=400)
