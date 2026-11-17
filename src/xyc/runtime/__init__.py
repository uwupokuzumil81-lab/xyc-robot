"""Model execution policy and robot-command orchestration.

This package contains no serial, ROS or model-training implementation.  It
connects a model adapter to an inference strategy and, optionally, to one
platform command sink.
"""

from .policy_executor import CommandResult, PolicyExecutor
from .policy_runner import PolicyRunner

__all__ = ['CommandResult', 'PolicyExecutor', 'PolicyRunner']
