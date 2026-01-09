"""Training Components Module

This module re-exports all components in the training category for easy import:
    from kfp_components.components.training import component_name
"""

from .finetuning import train_model

__all__ = ["train_model"]
