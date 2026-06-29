Installation
============

Requirements
------------

- Python 3.11
- Poetry
- Git
- Git LFS
- Docker and Docker Compose (for containerised deployment)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-researchgroup2.git
   cd 2025-26d-fai2-adsai-group-researchgroup2
   poetry env use python3.11
   poetry install

Verify the installation
-----------------------

.. code-block:: bash

   poetry run python -c "import sign_language; print('OK')"
   poetry run sign-language --help

Environment configuration
--------------------------

Copy the example environment file and fill in the required values:

.. code-block:: bash

   cp env.example .env

Key variables:

.. code-block:: text

   # Database
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/signlang
   SECRET_KEY=your-secret-key-at-least-32-characters

   # MLflow (optional)
   MLFLOW_ENABLED=false
   MLFLOW_TRACKING_URI=file:./logs/mlflow

   # Azure ML (optional, required for cloud training)
   AZURE_SUBSCRIPTION_ID=
   AZURE_RESOURCE_GROUP=
   AZURE_WORKSPACE=

Database setup
--------------

Run Alembic migrations before starting the server:

.. code-block:: bash

   poetry run alembic upgrade head

Docker deployment
-----------------

Start all services with Docker Compose:

.. code-block:: bash

   docker compose up --build

This starts the FastAPI backend, frontend, and supporting services
as defined in ``docker-compose.yml``.
