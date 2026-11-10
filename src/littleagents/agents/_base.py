"""
Base agent implementation following the stub.md specification.
This module provides the core BaseAgent class that all agents must inherit
from, implementing the interface specified in stub.md with proper typing and
functionality.
"""
# Imports tools for defining abstract base classes and abstract methods
from abc import ABC, abstractmethod
# Imports the async-generator type used in the streaming method annotation
from collections.abc import AsyncGenerator
# Imports the datetime type
from datetime import datetime
# Imports type hints used throughout the module
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

# Import Pydantic's base model for configuration and structured output types
from pydantic import BaseModel

# Imports the token used to request cancellation
from .._cancellation_token import CancellationToken
# Imports the base class for configurable components
from .._component_config import ComponentBase
# Imports the middleware types and their processing chain
from .._middleware import BaseMiddleware, MiddlewareChain
# Imports the agent's conversation and session context
from ..context import AgentContext
# Imports the strategy type for reducing or trimming context
from ..compaction import CompactionStrategy
# Imports the interface for chat-completion modle clients
from ..llm import BaseChatCompletionClient
# Imports the interface for agent memory
from ..memory import BaseMemory
# Imports message types used by the agent
from ..messages import Message, SystemMessage, UserMessage
# Imports the base tool and functional-tool types
from ..tools import BaseTool, FunctionalTool
# Imports event, response, chunk, and usage types
from ..types import AgentEvent, AgentResponse, ChatCompletionChunk, Usage

# Runs this block only for static type checking, avoiding runtime import
if TYPE_CHECKING:
    # Imports the wrapper type used in the as_tool() return annotation
    from ._agent_as_tool import AgentAsTool


# Creates a unique sentinel to distinguish omitted arguments from explicit values
_MISSING: Any = object()
"""Sentinel so a stale positional call is diagnosed before Python's generic error."""


# Defines an abstract agent class with Pydantic-based component configuration
class BaseAgent(ComponentBase[BaseModel], ABC):
    """
    Abstract base class defining the core agent interface.
    All agents in the littleagents framework must inherit from this base class
    and implement its abstract methods, following the stub.md specification.
    """
    # Defines the initializer for configuring a new agent
    def __init__(
        # Refers to the instance being initiali\ized
        self,
        name: str,
        # Captures extra positional arguments so outdated calls can be diagnosed well
        *_positional: Any,
        instructions: str = _MISSING,
        model_client: BaseChatCompletionClient = _MISSING,
        description: str = "",
        # Accepts optional tool objects or callable functions
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        memory: Optional[BaseMemory] = None,
        context: Optional[AgentContext] = None,
        middlewares: Optional[List[BaseMiddleware]] = None,
        max_iterations: int = 10,
        output_format: Optional[Type[BaseModel]] = None,
        # Controls whether tool results are followed by a LLM summary
        summarize_tool_result: bool = True,
        required_tools: Optional[List[str]] = None,
        # Provides example tasks for discovering the agent's capabilities
        example_tasks: Optional[List[str]] = None,
        compaction: Optional[CompactionStrategy] = None,
        # Accepts hooks that run before the first model call
        start_hooks: Optional[List] = None,
        # Accepts hooks that run when the agent would otherwise stop
        end_hooks: Optional[List] = None,
        # Captures additional keyword configuration
        **kwargs: Any
    ):
        """
        Initialize the base agent following stub.md specification.
        Args:
        - name: Unique identifier for the agent
        - instructions: Internal system prompt/role definition for LLM calls
        - model_client: Abstraction for LLM API calls
        - description: External-facing description for orchestrators/other agents
        - tools: Available tools for the agent
        - memory: Persistent storage for agent state
        - context: Agent context containing messages and metadata
        - middlewares: List of middleware to process operations
        - max_iterations: Maximum tool call iterations to prevent loops
        - output_format: Optional Pydantic model for structured output
        - summarize_tool_result: If False, agent stops after tool execution without LLM summarization
        - required_tools: Optional list of tool names that MUST be used (forced tool use)
        - example_tasks: Optional list of example tasks to help users discover agent capabilities
        - compaction: Optional compaction strategy for context management during tool loops
        When provided, the strategy's compact() method is called BEFORE each LLM call in the tool loop, allowing it to compact/trim messages.
        The compacted list replaces the working list for subsequent iterations.

        - start_hooks: Optional list of BaseStartHook instances (run deterministically before the first LLM call)
        - end_hooks: Optional list of BaseEndHook instances (run deterministically when the agent would stop and can inject a UserMessage to resume the loop)
        - **kwargs: Additional configuration
        """
        # Builds a list of required keyword arguments that were not supplied
        missing = [
            # Adds the argument name to the missing-argument list
            label
            # Iterates over each required argument name and its value
            for label, value in (
                # Chekcs whether instructions were omitted
                ("instructions", instructions),
                # Checks whether the model client was omitted
                ("model_client", model_client)
            )
            # Keeps only arguments still set to the sentinel
            if value is _MISSING
        ]
        # Checks whether any required keyword arguments were omitted
        if missing:
            # Raises a type error to explain the missing arguments
            raise TypeError(
                f"Agent(...) missing required keyword argument(s): "
                f"{', '.join(missing)}"
            )

        self.name = name
        self.description = description
        self.instructions = instructions
        self.model_client = model_client
        self.tools: List[BaseTool] = self._process_tools(tools or [])
        self.memory = memory
        self.context = context or AgentContext()
        self.middleware_chain = MiddlewareChain(middlewares)
        self.max_iterations = max_iterations
        self.output_format = output_format
        self.summarize_tool_result = summarize_tool_result
        self.required_tools = required_tools or []
        self.example_tasks = example_tasks or []
        self.compaction = compaction
        self.start_hooks = start_hooks or []
        self.end_hooks = end_hooks or []

        # Validate configuration
        # Checks that required agent settings are valid
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate agent configuration."""
        # Checks that the name is a nonempty stirng
        if not self.name or not isinstance(self.name, str):
            raise AgentConfigurationError("Agent name must be a non-empty string")

        # Checks that instructions were provided and are nonempty
        if not self.instructions:
            raise AgentConfigurationError("Agent instructions cannot be empty")

        # Checks that a model client is present
        if self.model_client is None:
            raise AgentConfigurationError("Model client is required")

    def _process_tools(self, tools: List[Union[BaseTool, Callable]]) -> List[BaseTool]:
        """
        Convert mixed tool types to BaseTool instances.
        Args:
        - tools: List of BaseTool instances or callable functions
        Returns:
        - List of BaseTool instances
        """
        # Creates a list to hold normalized tool objects
        processed = []
        # Iterates over each supplied tool
        for tool in tools:
            # Checks whether the tool already has the expected base type
            if isinstance(tool, BaseTool):
                # Keeps an existing BaseTool unchanged
                processed.append(tool)
            # Checks whether the item can be called like a function
            elif callable(tool):
                # Wraps the callable in a tool adapter
                processed.append(FunctionTool(tool))
            # Handles items that are neither tool objects nor callables
            else:
                # Raises a configuration error for an unsupported tool
                raise AgentConfigurationError(
                    f"Invalid tool type: {type(tool)}. Must be BaseTool or callable."
                )
        # Returns the normalized tool list
        return processed

    def _find_tool(self, name: str) -> Optional[BaseTool]:
        """
        Find tool by name.
        Args:
        - name: Tool name to search for
        Returns:
        - Tool instance or None if not found
        """
        return next((tool for tool in self.tools if tool.name == name), None)

    def _get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Convert tools to OpenAI function calling format.
        Returns:
        - List of tools in OpenAI function format
        """
        return [tool.to_llm_format() for tool in self.tools]

    # Requires conrete subclasses to implement this method
    @abstractmethod
    async def run(
        # Refers to the current agent instance
        self,
        # Accepts a string, user message, or list of messages as the task
        task: Union[str, UserMessage, List[Message]],
        # Optionally accepts a token for cancelling execution
        cancellation_token: Optional[CancellationToken] = None,
        # Controls whether run data should be persisted
        persist: bool = False
    ) -> AgentResponse:
        """
        Execute the agent's main reasoning and action loop.
        Args:
        - task: The task or query for the agent to address
        - cancellation_token: Optional token for cancelling execution
        - persist: If True, save the run to ~/.littleagents/ (DB index + JSON file with full response data)
        Returns:
        - AgentResponse containing messages and usage statistics
        Raises:
        - AgentError: If the agent encounters an error during execution
        - asyncio.CancelledError: If execution is cancelled
        """
        # Leaves the implementation to conrete subclasses
        pass

    # Requires concrete subclasses to implement this method
    @abstractmethod
    def run_stream(
        # Refers to the current agent instance
        self,
        # Acceps a string, user message, or list of messages as the task
        task: Union[str, UserMessage, List[Message]],
        # Optionally accepts a token for cancelling execution
        cancellation_token: Optional[CancellationToken] = None,
        # Controls whether detailed agent events are emitted
        verbose: bool = False,
    ) -> AsyncGenerator[
        # Lists the types the generator may yield
        Union[Message, AgentEvent, AgentResponse, ChatCompletionChunk], None]:
        """
        Execute the agent with streaming output.
        Args:
        - task: The task or query for the agent to address
        - cancellation_token: Optional token for cancelling execution
        - verbose: If True, emit agent events; if False, only emit messages and response
        - stream_tokens: If True, stream individual token chunks from LLM (implementation dependent)
        Yields:
        - Messages, events (if verbose=True), ChatCompletionChunks (if stream_tokens=True), and final AgentResponse with usage stats
        Raises:
        - AgentError: If the agent encounters an error during execution
        - asyncio.CancelledError: If execution is cancelled
        """
        # Leaves the implementation to concrete subclasses
        pass

    def _convert_task_to_messages(
        self, task: Union[str, UserMessage, List[Message]]
    ) -> List[Message]:
        """
        Convert task input to proper message format.
        Args:
        - task: Task in various formats
        Returns:
        - List of messages
        """
        # Checks whether the task is plain text
        if isinstance(task, str):
            # Wraps the text in a user message
            return [UserMessage(content=task, source="user")]
        # Checks whether the task is already a user message
        elif isinstance(task, UserMessage):
            # Returns that message in a one-item list
            return [task]
        # Checks whether the task is already a list of messages
        elif isinstance(task, list):
            # Returns the supplied message list unchanged
            return task
        else:
            # Reports the unsupported input type
            raise AgentExecutionError(f"Unsupported task type: {type(task)}")

    async def _prepare_llm_messages(
        self,
        task_messages: List[Message],
        context: Optional["AgentContext"] = None,
    ) -> List[Message]:
        """
        Prepare messages for LLM call including system instructions, memory context, and history.
        Args:
        - task_messages: Messages from the current task
        - context: Explicit context to use. If None, falls back to self.context. Passing context explicitly makes this method safe for concurrent use.
        Returns:
        - Complete list of messages for LLM
        """
        working_context = context if context is not None else self.context
        messages = []

        # Add system message with instructions
        system_content = self.instructions

        # Add forced tool use instruction if required_tools is set
        if self.required_tools:
            tool_list = ", ".join(self.required_tools)
            system_content += f"\n\nIMPORTANT: You MUST use these tools in your response: {tool_list}. Do not respond without calling these tools."

        # Add memory context if available
        if self.memory:
            try:
                # Get relevant context from memory based on current task
                current_task = task_messages[0].content if task_messages else ""
                memory_result = await self.memory.get_context(max_items=5)

                # Handle both old (List[str]) and new (MemoryQueryResult) interfaces
                if hasattr(memory_result, "results"):
                    # New MemoryQueryResult interface
                    context_items = []
                    for memory_content in memory_result.results:
                        if isinstance(memory_content.content, str):
                            context_items.append(memory_content.content)
                        else:
                            # Handle dict/json content
                            import json

                            context_items.append(json.dumps(memory_content.content))
                    context_items_list = context_items
                else:
                    # Legacy List[str] interface (backward compatibility)
                    context_items_list = memory_result

                if context_items_list and isinstance(context_items_list, list):
                    system_content += "\n\nRelevant context from memory:\n" + "\n".join(
                        context_items_list
                    )
            except Exception:
                # Don't fail if memory access fails
                pass
        # Inject dynamic sections from tools that provide them (e.g., SkillsTool)
        for tool in self.tools:
            if hasattr(tool, "get_system_prompt_section"):
                try:
                    section = tool.get_system_prompt_section()
                    if section:
                        system_content += section
                except Exception:
                    pass


        messages.append(SystemMessage(content=system_content, source="system"))

        # Add message history from context
        messages.extend(working_context.messages)

        # Add current task message
        messages.extend(task_messages)

        return messages

    async def reset(self) -> None:
        """
        Reset the agent to a clean state.
        Clears conversation history and temporary state while preserving core configuration.
        """
        self.context.reset()

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the agent for debugging and coordination.
        Returns:
        - Dictionary containing agent metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
            "model": getattr(self.model_client, "model", "unknown"),
            "tools_count": len(self.tools),
            "has_memory": self.memory is not None,
            "has_middlewares": len(self.middleware_chain.middlewares) > 0,
            "message_history_length": self.context.message_count
        }

    def get_conversation_data(self) -> Dict[str, Any]:
        """
        Get current conversation data for application-managed memory state.
        Returns:
        - Dictionary containing conversation context that applications can use to decide what to store in memory
        """
        # Imports the specific message types needed for categorization
        from ..messages import AssistantMessage, ToolMessage, UserMessage

        user_messages = [
            # Selects only messages authored as user messages
            msg for msg in self.context.messages if isinstance(msg, UserMessage)
        ]
        assistant_messages = [
            # Selects only assistant messages
            msg for msg in self.context.messages if isinstance(msg, AssistantMessage)
        ]
        tool_messages = [
            # Selects only tool messages
            msg for msg in self.context.messages if isinstance(msg, ToolMessage)
        ]

        return {
            "agent_name": self.name,
            "total_messages": self.context.message_count,
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "tool_messages": len(tool_messages),
            "tools_used": list(
                set([msg.tool_name for msg in tool_messages if msg.success])
            ),
            "last_user_message": user_messages[-1].content if user_messages else None,
            "last_assistant_message": assistant_messages[-1].content
            if assistant_messages
            else None,
            "session_id": self.context.session_id,
            "metadata": self.context.metadata,
            "conversation_history": [
                {
                    "type": type(msg).__name__,
                    "content": ms.content[:200] + "..."
                    if len(msg.content) > 200
                    else msg.content,
                    "timestamp": getattr(msg, "timestamp", None),
                }
                for msg in self.context.messages
            ],
        }

    def as_tool(
        self,
        task_parameter_name: str = "task",
        result_strategy: "Union[str, Callable[List[Message]], str]]" = "last",
    ) -> "AgentAsTool":
        """
        Converts this agent into a tool that other agents can use.
        This enables hierarchical composition where specialized agents can be used as tools by higher-level coordinating agents.
        Args:
        - task_parameter_name: Parameter name for the task input (default: "task")
        - result_strategy: Strategy for extracting result from messages. Can be "last" (default), "last:N", "all", or a callable that takes a list of messages and returns a string.
        Returns:
        - AgentAsTool instance wrapping this agent
        """
        # Imports the wrapper locally to avoid an import cycle
        from ._agent_as_tool import AgentAsTool

        # Creates and returns the tool wrapper
        return AgentAsTool(self, task_parameter_name, result_strategy)

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation for the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description[:50]}...')"

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager with cleanup."""
        # Subclasses can override to add specific cleanup
        # Default behavior: no cleanup needed for base agent
        return False


class AgentError(Exception):
    """Base exception class for agent-related errors."""
    def __init__(self, message: str, agent_name: Optional[str] = None):
        self.agent_name = agent_name
        super().__init__(message)

    def __str__(self) -> str:
        if self.agent_name:
            return f"Agent '{self.agent_name}': {super().__str__()}"


class AgentExecutionError(AgentError):
    """Raised when an agent encounters an error during task execution."""
    pass


class AgentConfigurationError(AgentError):
    """Raised when an agent is misconfigured."""
    pass


class AgentToolError(AgentError):
    """Raised when an agent's tool execution fails."""
    pass


class AgentMemoryError(AgentError):
    """Raised when agent memory operations fail."""
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent execution times out."""
    pass
