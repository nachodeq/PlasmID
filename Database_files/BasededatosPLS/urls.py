# queries/urls.py

from django.urls import path
from miapBBDDpls import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('queries/', views.queries_home, name='queries_home'),
    path('queries/premade/', views.premade_queries, name='premade_queries'),
    path('queries/save/', views.save_queries, name='save_queries'), 
    path('queries/schema/', views.database_schema, name='database_schema'),  
    path('queries/execute/', views.execute_query, name='execute_query'),
    path('natural_language_query/', views.natural_language_query, name='natural_language_query'),
    path('download-csv/', views.download_csv, name='download_csv'),  
    path('queries/new/', views.new_query, name='new_query'),
    path('examples/', views.examples, name='examples'),
]

