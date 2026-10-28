"""
Middleware system for littleagents.

This module provides the middleware infrastructure for intercepting and
processing agent operations like model calls, tool calls, and memory access.

Middleware uses async generators to support:
- Event emission (for observability)
- Approval requests (pausing execution)
- Streaming transformations
- Error recovery
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .context import AgentContext
from .messages import Message

class BaseMiddleware(ABC):
    """Abstract base class for middleware."""

    @abstractmethod
    async def process_requests(
        self,
        context: MiddlewareContext
    ) -> MiddlewareContext:
        """Process before the operation executes."""
        return context

    @abstractmethod
    async def process_response(
        self,
        context: MiddlewareContext,
        result: Any
    ) -> Any:
        """Process after the operation completes successfully."""
        return result

    @abstractmethod
    async def process_error(
        self,
        context: MiddlewareContext,
        error: Exception
    ) -> Optional[Any]:
        """Handle errors from the operation."""
        raise error

class SecurityMiddleware(BaseMiddleware):
    """Blocks malicious input before it reaches the model."""

    def __init__(self):
        self.malicious_patterns = [
            r"ignore.*previous.*instructions",
            r"system.*prompt.*injection",
            r"\\x[0-9a-f]{2}", # Hex encoding attempts
            r"<script.*?>.*?</script>", # Script injection
        ]

    async def process_request(
        self,
        context: MiddlewareContext
    ) -> MiddlewareContext:
        """Block malicious requests before they reach the model."""
        if context.operation == "model_call":
            for message in context.data:
                if hasattr(message, "content"):
                    for pattern in self.malicious_patterns:
                        if re.search(
                            pattern,
                            message.content,
                            re.IGNORECASE
                        ):
                            raise ValueError(
                                "Blocked potentially malicious input"
                            )
        return context

class LoggingMiddleware(BaseMiddleware):
    """Log all agent operations using before/after hooks."""

    async def process_request(
        self,
        context: MiddlewareContext
    ) -> MiddlewareContext:
        """Log operation start."""
        print(f"[{context.agent_name}] Starting {context.operation}")
        context.metadata["start_time"] = time.time()
        return context

    async def process_response(
        self,
        context: MiddlewareContext,
        result: Any
    ) -> Any:
        """Log successful completion."""
        duration = time.time() - context.metadata.get("start_time", 0)
        print(f"[{context.agent_name}] \
            {context.operation} completed in {duration:.2f}s")
        return result

    async def process_error(
        self,
        context: MiddlewareContext,
        error: Exception
    ) -> Optional[Any]:
        """Log operation failures."""
        print(f"[{context.agent_name}] {context.operation} failed: {error}")
        raise error # Re-raise the error


