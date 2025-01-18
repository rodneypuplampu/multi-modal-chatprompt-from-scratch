# models.py
from django.db import models
from django.contrib.auth.models import User

class ChatRoom(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class Message(models.Model):
    room = models.ForeignKey(ChatRoom, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    media_file = models.FileField(upload_to='chat_media/', null=True, blank=True)
    media_type = models.CharField(max_length=20, choices=[
        ('image', 'Image'),
        ('video', 'Video'),
        ('text', 'Text')
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    dlp_analyzed = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username}: {self.content[:50]}"
