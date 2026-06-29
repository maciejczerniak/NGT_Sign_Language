Usage
=====

CLI
---

The package provides a ``sign-language`` command with four subcommands.

Show help
~~~~~~~~~

.. code-block:: bash

   poetry run sign-language --help
   poetry run sign-language predict --help
   poetry run sign-language serve --help
   poetry run sign-language train --help

Run inference on an image
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run sign-language predict --image path/to/hand.jpg

With custom model checkpoints:

.. code-block:: bash

   poetry run sign-language predict --image path/to/hand.jpg \
       --model models/best_ngt_model_v2.pth \
       --lm-model models/best_landmark_mlp.pth \
       --landmarker hand_landmarker.task

Additional options:

.. code-block:: bash

   poetry run sign-language predict --image hand.jpg --top-k 3 --verbose

Start the API server
~~~~~~~~~~~~~~~~~~~~

Development mode (auto-reload on file changes):

.. code-block:: bash

   poetry run sign-language serve --reload

Production mode with multiple workers:

.. code-block:: bash

   poetry run sign-language serve --host 0.0.0.0 --port 8000 --workers 4

Run the training workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune EfficientNet-B0 on the NGT dataset:

.. code-block:: bash

   poetry run sign-language train \
       --pretrained-checkpoint models/best_ngt_model_v2.pth \
       --data-dir data/raw \
       --epochs 30 \
       --batch-size 16

Run the local retraining pipeline (preprocess → train):

.. code-block:: bash

   poetry run sign-language local-pipeline \
       --pretrained-checkpoint models/best_ngt_model_v2.pth \
       --raw-data-dir data/raw \
       --epochs 5 \
       --batch-size 8

API
---

Once the server is running, interactive API documentation is available at:

- http://127.0.0.1:8000/docs — Swagger UI
- http://127.0.0.1:8000/redoc — ReDoc

HTTP endpoints
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - Method
     - Endpoint
     - Description
   * - GET
     - ``/api/health``
     - Liveness check. Returns ``{"status": "ok"}``.
   * - GET
     - ``/api/info``
     - Returns app version, device, class count, and model availability.
   * - POST
     - ``/api/predict``
     - Run inference on a single base64-encoded image frame.
   * - POST
     - ``/api/reset``
     - Reset the prediction smoother and sequence builder.
   * - POST
     - ``/api/auth/jwt/login``
     - Obtain a JWT bearer token.
   * - POST
     - ``/api/auth/register``
     - Register a new user account.

Send a prediction request:

.. code-block:: bash

   curl -X POST http://127.0.0.1:8000/api/predict \
       -H "Content-Type: application/json" \
       -d '{"image": "<base64-encoded-frame>"}'

WebSocket endpoint
~~~~~~~~~~~~~~~~~~

Connect to ``ws://127.0.0.1:8000/ws/predict`` for real-time multi-hand
prediction over a persistent connection.

Send a frame:

.. code-block:: json

   {"image": "<base64-encoded-frame>"}

Reset the server-side state:

.. code-block:: json

   {"action": "reset"}

Training trigger API
--------------------

Start the trigger service:

.. code-block:: bash

   poetry run uvicorn sign_language_training.trigger_api.app:app \
       --host 0.0.0.0 --port 8010

Health check:

.. code-block:: bash

   curl http://localhost:8010/health

Trigger a manual retraining job:

.. code-block:: bash

   curl -X POST http://localhost:8010/train \
       -H "X-API-Key: your-api-key" \
       -H "Content-Type: application/json" \
       -d '{"reason": "manual", "force": true}'

Azure ML pipeline submission
-----------------------------

Submit the preprocessing and training pipeline:

.. code-block:: bash

   poetry run python scripts/submit_pipeline.py

Submit a hyperparameter sweep:

.. code-block:: bash

   poetry run python scripts/submit_sweep_job.py \
       --max-total-trials 12 \
       --batch-sizes 8,16,32 \
       --patience-values 5,7,10
