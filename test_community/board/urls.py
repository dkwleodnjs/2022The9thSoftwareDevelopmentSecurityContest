from django.urls import path
from . import views

urlpatterns = [
    path('detail/<int:pk>/', views.board_detail),
    path('detail/<int:pk>/delete/', views.board_delete),
    path('detail/<int:pk>/update/', views.board_update),
    path('list/', views.board_list),
    path('write/', views.board_write)
]
