"""Verifier-first synthesis of grounded long-horizon computer-use tasks."""

from derail.synthesis.compatibility import TypeSystem, build_compatibility_edges
from derail.synthesis.graph import (
    SynthesisValidationError,
    compose_modules,
    compute_complexity,
    validate_grounded_module,
    validate_task_fragment,
)
from derail.synthesis.pipeline import SynthesisConfig, SynthesisPipeline, SynthesisResult
from derail.synthesis.sampling import compute_skeleton_weights
from derail.synthesis.skeletons import (
    canonical_structural_signature,
    extract_observed_skeletons,
)

__all__ = [
    "SynthesisConfig",
    "SynthesisPipeline",
    "SynthesisResult",
    "SynthesisValidationError",
    "TypeSystem",
    "build_compatibility_edges",
    "canonical_structural_signature",
    "compose_modules",
    "compute_complexity",
    "compute_skeleton_weights",
    "extract_observed_skeletons",
    "validate_grounded_module",
    "validate_task_fragment",
]
