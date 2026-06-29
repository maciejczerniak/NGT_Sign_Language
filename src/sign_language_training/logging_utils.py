"""Logging helpers for the training workflow.

Provides a single summary logger called at the end of a training run
to print paths, dataset stats, model parameters, and evaluation metrics
in a consistent format.
"""

from __future__ import annotations

import logging

from sign_language_training.configuration import TrainingConfig, TrainingPaths
from sign_language_training.model_evaluation import EvaluationSummary
from sign_language_training.model_training import TrainingResult

logger = logging.getLogger(__name__)


def log_training_summary(
    paths: TrainingPaths,
    config: TrainingConfig,
    dataset_size: int,
    num_classes: int,
    trainable_parameters: int,
    total_parameters: int,
    training_result: TrainingResult,
    evaluation_summary: EvaluationSummary,
) -> None:
    """Log the final training and evaluation summary for a workflow run.

    Prints a formatted block to the logger covering the checkpoint name,
    dataset size, validation split, parameter counts, epoch count, best
    validation accuracy, all evaluation metrics, and the path to the saved
    checkpoint.

    Args:
        paths: :class:`~sign_language_training.configuration.TrainingPaths`
            providing the pretrained checkpoint name and best model path.
        config: :class:`~sign_language_training.configuration.TrainingConfig`
            providing target accuracy, validation split ratio, and random seed.
        dataset_size: Total number of images in the dataset.
        num_classes: Number of output classes.
        trainable_parameters: Number of trainable model parameters.
        total_parameters: Total number of model parameters including
            frozen layers.
        training_result: :class:`~sign_language_training.model_training.TrainingResult`
            containing best validation accuracy and epochs trained.
        evaluation_summary: :class:`~sign_language_training.model_evaluation.EvaluationSummary`
            containing accuracy, F1, precision, and recall metrics.
    """
    target_status = (
        "TARGET MET"
        if training_result.best_val_accuracy >= config.target_accuracy
        else "BELOW TARGET"
    )
    logger.info("%s", "=" * 60)
    logger.info("NGT fine-tuning summary")
    logger.info("Pretrained checkpoint: %s", paths.pretrained_checkpoint.name)
    logger.info("Dataset size: %d images across %d classes", dataset_size, num_classes)
    logger.info(
        "Validation split: %.0f%% (seed=%d)", config.val_split * 100, config.seed
    )
    logger.info(
        "Trainable parameters: %d / %d",
        trainable_parameters,
        total_parameters,
    )
    logger.info("Epochs trained: %d", training_result.epochs_trained)
    logger.info(
        "Best validation accuracy: %.4f (%s)",
        training_result.best_val_accuracy,
        target_status,
    )
    logger.info("Accuracy: %.4f", evaluation_summary.accuracy)
    logger.info("Macro F1: %.4f", evaluation_summary.f1_macro)
    logger.info("Weighted F1: %.4f", evaluation_summary.f1_weighted)
    logger.info("Macro Precision: %.4f", evaluation_summary.precision_macro)
    logger.info("Macro Recall: %.4f", evaluation_summary.recall_macro)
    logger.info("Best checkpoint: %s", paths.best_model_path)
    logger.info("%s", "=" * 60)
