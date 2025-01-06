## PlasmID 
Master thesis Ignacio de Quinto, Máster en bioinformática y biología computacional. Universidad Autónoma de Madrid, 2024-2025

PlasmID is a hierarchical plasmid database designed to streamline the study of plasmids, antibiotic resistance, and horizontal gene transfer. The project provides a Django-based web interface for querying plasmid data and integrates tools for annotation and natural language queries.



Repository Structure

PlasmID/
├── few_shot_examples.json       # JSON file containing examples for LLM queries
├── database_build.py            # Script for constructing and populating the database
├── Database_files/              # Main directory for the Django project
│   ├── manage.py                # Django management script
│   ├── db.sqlite3               # SQLite database file
│   ├── miappBBDDpls/            # Django application
│   │   ├── __init__.py
│   │   ├── admin.py             # Django admin configuration
│   │   ├── apps.py              # Django app configuration
│   │   ├── models.py            # Database models for the application
│   │   ├── tests.py             # Unit tests for the application
│   │   ├── views.py             # View logic for handling web requests
│   │   ├── templates/           # HTML templates for the web interface
│   │   │   ├── base.html        # Base template for the project
│   │   │   ├── database_schema.html  # Displays database schema information
│   │   │   └── ...              # Other HTML templates
│   ├── BasededatosPLS/          # Django project configuration
│   │   ├── __init__.py
│   │   ├── asgi.py              # ASGI configuration
│   │   ├── settings.py          # Project settings
│   │   ├── urls.py              # URL routing configuration
│   │   ├── wsgi.py              # WSGI configuration
└── ...
