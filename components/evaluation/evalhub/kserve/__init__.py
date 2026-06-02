"""Eval Hub Evaluation Component — KServe variant.

This component provides model evaluation via the Eval Hub service,
using a KServe InferenceService + ServingRuntime for model serving
(matching the RHOAI dashboard deployment pattern).
"""

from .component import evalhub_evaluator_kserve

__all__ = ["evalhub_evaluator_kserve"]
