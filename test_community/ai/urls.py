from django.urls import path
from . import views

urlpatterns = [
    path('learnedgesture/', views.learned_model_gesture),
    path('gesture/', views.gesture_register),
    path('authentication/', views.authentication_model_gesture ),
    path('initmonitor/', views.Initdetectme ),
    path('learnedmonitor/', views.learneddetectme ),
    path('learningmonitor/', views.learningdetectme ),
    path('authenticationmonitor/', views.authenticationdetectme ),
]
