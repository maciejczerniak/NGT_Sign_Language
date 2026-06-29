Sign Language Recognition
=========================

A real-time Dutch Sign Language (NGT) recognition system built with
EfficientNet-B0, MediaPipe, and FastAPI.

The project provides:

- A **FastAPI backend** with REST and WebSocket endpoints for real-time
  per-frame prediction.
- A **CLI** for running inference, starting the server, and training.
- A **training package** (``sign_language_training``) for fine-tuning,
  augmentation, and Azure ML pipeline submission.
- An **MLOps layer** with MLflow tracking, Azure ML SDK v2 integration,
  and automated retraining triggers.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   reference/sign_language
   reference/sign_language_training
   reference/sign_language_azure_api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
