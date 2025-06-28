from django.urls import path
from .views import (
    IngestDataView, 
    ChatAPIView,
)
urlpatterns = [
    path('ingest/', IngestDataView.as_view(), name='ingest-data'),
    path('chat/', ChatAPIView.as_view(), name='chat'),
]