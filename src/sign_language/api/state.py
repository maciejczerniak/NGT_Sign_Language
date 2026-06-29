"""Application-wide state: loaded models + stateful helpers."""

from dataclasses import dataclass, field
from threading import Lock

from sign_language.utils.smoothing import PredictionSmoother
from sign_language.utils.sequence import SequenceBuilder
from sign_language.models.loader import LoadedModels


@dataclass
class AppState:
    """Shared application state holding loaded models and stateful inference helpers.

    One instance is created at startup in the lifespan context and stored on
    ``app.state.app_state``. All mutable fields are protected by ``lock`` for
    thread-safe access from concurrent requests.

    :param models: The loaded ML models and associated metadata, populated
        at startup by :func:`~sign_language.models.loader.load_all`.
    :param smoother: Prediction smoother that stabilises per-frame letter
        predictions across consecutive frames.
    :param sequence: Sequence builder that accumulates committed letters into
        words and sentences.
    :param lock: Threading lock protecting ``smoother`` and ``sequence``
        from concurrent modification.
    """

    models: LoadedModels
    smoother: PredictionSmoother = field(default_factory=PredictionSmoother)
    sequence: SequenceBuilder = field(default_factory=SequenceBuilder)
    lock: Lock = field(default_factory=Lock)

    def reset(self) -> None:
        """Reset the smoother and sequence builder to their initial state.

        Acquires ``lock`` before clearing both stateful helpers to ensure
        thread safety during concurrent request handling.
        """
        with self.lock:
            self.smoother.clear()
            self.sequence.clear()
