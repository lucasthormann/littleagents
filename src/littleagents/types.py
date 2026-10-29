"""
Core data types and models for the picoagents framework using Pydantic.

This module defines all the structured types used throughout the framework for type safety and data validation.
"""

from datetime import datetime
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

from .messages import Message

if TYPE_CHECKING:
    from .context import AgentContext, ToolApprovalRequest


