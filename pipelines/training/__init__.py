"""Training Pipelines Module

This module re-exports all pipelines in the training category for easy import:
    from kfp_components.pipelines.training import pipeline_name
"""

from .sft import sft_pipeline
from .osft import osft_pipeline

__all__ = ["sft_pipeline", "osft_pipeline"]
