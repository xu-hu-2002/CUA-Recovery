"""Agent interface layer between DERAIL and the frozen MyPCBench runner."""

from .factory import create_mypcbench_agent

__all__ = ["create_mypcbench_agent"]
