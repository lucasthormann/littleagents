"""
Base agent implementation following the stub.md specification.
This module provides the core BaseAgent class that all agents must inherit
from, implementing the interface specified in stub.md with proper typing and
functionality.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type,
Union

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



class BaseAgent(ComponentBase[BaseModel], ABC):
    """
    Abstract base class defining the core agent interface.

    All agents in the littleagents framework must inherit from this base class
    and implement its abstract methods, following the stub.md specification.
    """

    name: str,
    instructions: str,
    model_client: 'BaseChatCompletionClient',
    tools: Optional[List] = None,
    memory: Optional['BaseMemory'] = None,
    context: Optional['AgentContext'] = None,
    middleware: Optional[List] = None,
    max_iterations: int = 10
):
    self.name = name
    self.instructions = instructions
    self.model_client = model_client

    # Process optional components with defaults
    self.tools = self._process_tools(tools or [])
    self.memory = memory
    self.context = context or AgentContext()
    self.middleware_chain = MiddlewareChain(
        middleware or []
    )

    @abstractmethod
    async def run(
        self,
        task: Union[str, UserMessage, List[Message]]
    ) -> 'AgentResponse':
        """Execute agent and return final response."""
        pass

    @abstractmethod
    def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]]
    ) -> AsyncGenerator[Union[Message, 'AgentEvent'], None]:
        """Execute agent with streaming output."""
        pass
