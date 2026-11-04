"""
Base agent implementation following the stub.md specification.
This module provides the core BaseAgent class that all agents must inherit
from, implementing the interface specified in stub.md with proper typing and
functionality.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from .._cancellation_token import CancellationToken
from .._component_config import ComponentBase
from .._middleware import BaseMiddleware, MiddlewareChain
from ..context import AgentContext
from ..compaction import CompactionStrategy
from ..llm import BaseChatCompletionClient
from ..memory import BaseMemory
from ..messages import Message, SystemMessage, UserMessage
from ..tools import BaseTool, FunctionalTool
from ..types import AgentEvent, AgentResponse, ChatCompletionChunk, Usage

if TYPE_CHECKING:
    from ._agent_as_tool import AgentAsTool


_MISSING: Any = object()
"""Sentinel so a stale positional call is diagnosed before Python's generic error."""


class BaseAgent(ComponentBase[BaseModel], ABC):
    """
    Abstract base class defining the core agent interface.
    All agents in the littleagents framework must inherit from this base class
    and implement its abstract methods, following the stub.md specification.
    """
    def __init__(
        self,
        name: str,
        *_positional: Any,
        instructions: str = _MISSING,
        model_client: BaseChatCompletionClient = _MISSING,
        description: str = "",
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        memory: Optional[BaseMemory] = None,
        context: Optional[AgentContext] = None,
        middlewares: Optional[List[BaseMiddleware]] = None,
        max_iterations: int = 10,
        output_format: Optional[Type[BaseModel]] = None,
        summarize_tool_result: bool = True,
        required_tools: Optional[List[str]] = None,
        example_tasks: Optional[List[str]] = None,
        compaction: Optional[CompactionStrategy] = None,
        start_hooks: Optional[List] = None,
        end_hooks: Optional[List] = None,
        **kwargs: Any
    ):
        """
        Initialize the base agent following stub.md specification.

