"""
Base tool classes and interfaces for the LittleAgents framework.
This module defines the abstract base classes and core functionality for tools that agents can use to interact with the world.
"""
# Provides utilities for inspecting function signatures and coroutine functions
import inspect
# Supports abstract base classes and abstract methods
from abc import ABC, abstractmethod
# Provides the base class for enumerations
from enum import Enum
# Imports type annotations and helpers
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_type_hints
)

# Imports Pydantic's base model for component configuration
from pydantic import BaseModel

# Imports the common base class for components
from .._component_config import ComponentBase
# Imports the result type returned by tool execution
from ..types import ToolResult

# Runs these imports only for type checkers, avoiding runtime import cycles
if TYPE_CHECKING:
    # Type used to request cancellation
    from .._cancellation_token import CancellationToken
    # Message type that streaming tools may yield
    from ..messages import Message
    # Event type that streaming tools may yield
    from ..types import AgentEvent


class ApprovalMode(Enum):
    """Tool approval requirements."""
    # Allows execution without requesting approval
    NEVER = "never_require"
    # Requires approval before execution
    ALWAYS = "always_require"


class BaseTool(ComponentBase[BaseModel], ABC):
    """
    Abstract base class that all tools must implement.
    Defines the interface for tools that agents can use to perform actions beyond text generation (e.g., web search, file operations, API calls).
    """
    # Initializes a tool's common configuration
    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        approval_mode: ApprovalMode = ApprovalMode.NEVER
    ):
        """
        Initialize the base tool.
        Args:
        - name: Unique tool identifier
        - description: What the tool does (for LLM understanding)
        - version: Tool version following semantic versioning (default: "1.0.0")
        - approval_mode: Whether approval is required before execution
        """
        self.name = name
        self.description = description
        self.version = version
        self.approval_mode = approval_mode

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        JSON schema defining expected inputs for this tool.
        Returns:
        - JSON schema dictionary describing tool parameters
        """
        # Leaves the implementation to subclasses
        pass

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given parameters.
        Args:
        - parameters: Tool input parameters
        Returns:
        - ToolResult: Containing execution outcome
        """
        # Leaves the implementation to subclasses
        pass

    async def execute_stream(
        self,
        parameters: Dict[str, Any],
        cancellation_token: Optional["CancellationToken"] = None
    ) -> AsyncGenerator[Union["Message", "AgentEvent", ToolResult], None]:
        """
        Execute the tool with streaming output support.
        Tools can override this to provide streaming capabilities. Default implementation wraps execute() for backward compatibility.
        Args:
        - parameters: Tool input parameters
        - cancellation_token: Optional token for cancelling execution
        Yields:
        - Messages
        - Events during execution
        - Final ToolResult
        """
        # Default implementation (just call execute and yield result)
        result = await self.execute(parameters)
        yield result

    # Reports whether the subclass implements streaming
    def support_streaming(self) -> bool:
        """
        Check if this tool supports streaming execution.
        Returns:
        - True if tool has overriden execute_stream and False otherwise
        """
        # Check if execute_stream was overriden by comparing the subclass method with the default
        return type(self).execute_stream is not BaseTool.execute_stream

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate that the provided parameters match the tool's schema.
        Args:
        - params: Parameters to validate
        Returns:
        - True if parameters are valid and False otherwise
        """
        try:
            # Basic validation (subclasses can override for more checks)
            schema = self.parameters
            required_fields = schema.get("required", [])

            # Check required fields are present
            for field in required_fields:
                if field not in params:
                    # Rejects the parameters if a required field is missing
                    return False

            # Check parameter types if specified in schema
            # Gets the schema's property definitions
            properties = schema.get("properties", {})
            # Checks each supplied parameter
            for param_name, param_value in params.items():
                # Validates types only for declared properties
                if param_name in properties:
                    # Reads the property's expected JSON type
                    expected_type = properties[param_name].get("type")
                    # Checks the value when a type is specified
                    if expected_type and not self._check_type(
                        # Passes the supplied value and expected type
                        param_value, expected_type
                    ):
                        # Rejects the parameters when a type does not match
                        return False
            # Accepts the parameters if all checks pass
            return True
        # Catches errors caused by invalid schema or unexpected values
        except Exception:
            return False

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type."""
        # Maps JSON schema type names to Python types        
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }

        # Looks up the Python type for the schema type
        expected_python_type = type_mapping.get(expected_type)
        # Handles schema types absent from the mapping
        if expected_python_type is None:
            return True

        # Checks whether the value has the expected Python type
        return isinstance(value, expected_python_type)

    # Converts the tool into a funciton-calling representation
    def to_llm_format(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI function calling format.
        Returns:
        - Dictionary in OpenAI function format
        """
        # Include version in tool name for compatibility tracking (Anthropic pattern)
        # Format: toolname_v1.0.0 for versioned tools
        versioned_name = (
            f"{self.name}_v{self.version}" if self.version != "1.0.0" else self.name
        )

        # Builds and returns the function-calling structure
        return {
            # Identifies the entry as a callable function
            "type": "function",
            # Holds the function's metadata and schema
            "function": {
                # Sets the exported function name
                "name": versioned_name,
                # Supplies the function description
                "description": self.description,
                # Supplies the JSON schema for its input
                "parameters": self.parameters
            },
        }

    # Defines the tool's user-friendly string representation
    def __str__(self) -> str:
        # Shows the class and tool name
        return f"{self.__class__.__name__}(name='{self.name}')"

    # Defines the tool's developer-oriented representation
    def __repr__(self) -> str:
        # Shows the class, name, and description
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"


# Implements a tool by wrapping a Python function
class FunctionTool(BaseTool):
    """
    Tool that wraps a Python function for use by agents.
    Automatically extracts function metadata (name, docstring, parameters) and provides parameter validation based on type hints.
    """
    # Initializes the wrapper and build its parameter schema
    def __init__(
        # Refers to the new FunctionTool instance
        self,
        # Supplies the function to wrap
        func: Callable,
        # Optionally overrides the function's name
        name: Optional[str] = None,
        # Optionally overrides the function's description
        description: Optional[str] = None,
        # Sets default tool version
        version: str = "1.0.0"
        # Sets default approval policy
        approval_mode: ApprovalMode = ApprovalMode.NEVER
    ):
        """
        Create a tool from a Python function.
        Args:
        - func: The function to wrap as a tool
        - name: Optional custom name (defaults to function name)
        - description: Optional custom description (defaults to function docstring)
        - version: Tool version following semantic versioning (default: "1.0.0")
        - approval_mode: Whether approval is required before execution
        """
        # Stores the wrapper function
        self.func = func
        # Uses the custom name or falls back to the function name
        tool_name = name or func.__name__
        # Uses the override, docstring, or fallback text
        tool_description = (
            description or func.__doc__ or f"Execute {func.__name__} function"
        )

        # Initializes the shared BaseTool fields
        super().__init__(tool_name, tool_description, version, approval_mode)

        # Extract function metadata
        # Captures the function's parameters and defaults
        self.signature = inspect.signature(func)
        # Resolves the function's type annotations
        self.type_hints = get_type_hints(func)
        # Creates the JSON schema for function inputs
        self._parameters_schema = self._build_parameters_schema()

    # Exposes the generated schema as a property
    @property
    # Declares the schema property's return type
    def parameters(self) -> Dict[str, Any]:
        """Get JSON schema for function parameters."""
        # Returns the schema built during initialization
        return self._parameters_schema

    # Allows the tool object to be called like its wrapped function
    def __call__(self, *args, **kwargs):
        """
        Make the tool callable like the original function.
        This allows decorated functions to be called directly:
            @tool
            def get_weather(city: str) -> str:
                return f"Weather in {city}"

            result = get_weather("Seattle")
        """
        # Passes positional and keyword arguments to the original function
        return self.func(*args, **kwargs)

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the wrapped function with validated parameters.
        Args:
        - parameters: Function arguments
        Returns:
        - ToolResult: With function execution outcome
        """
        # Catches errors from validation or function execution
        try:
            # Validate parameters before execution by checking the inputs against the generated schema
            if not self.validate_parameters(parameters):
                # Returns a failure result for invalid inputs                
                return ToolResult(
                    success=False,
                    result=None,
                    error="Invalid parameters provided",
                    metadata={"tool_name": self.name}
                )

            # Execute the function
            # Checks whether the wrapped function is asynchronous
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(**parameters)
            # Handles a synchronous function
            else:
                result = self.func(**parameters)

            # Wraps the function's output in a successful tool result
            return ToolResult(
                success=True,
                # Stores the function's return value
                result=result,
                error=None,
                # Identifies the tool in the result metadata
                metadata={"tool_name": self.name}
            )

        # Catches exceptions raised during execution
        except Exception as e:
            # Converts the exception into a failure result
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                # Records the tool and exception type
                metadata={"tool_name": self.name, "exception_type": type(e).__name__}
            )

    def _build_parameters_schema(self) -> Dict[str, Any]:
        """
        Build JSON schema from function signature and type hints.
        Returns:
        - JSON schema dictionary
        """
        # Starts an object schema with no fields
        schema = {"type": "object", "properties": {}, "required": []}

        # Process each parameter in the function signature
        for param_name, param in self.signature.parameters.items():
            # Skip 'self' parameter since its not an input to the wrapped function
            if param_name == "self":
                continue

            # Determine parameter type
            # Looks up the parameter's type annotations
            param_type = self.type_hints.get(param_name)
            # Converts the annotation to a JSON schema type
            json_type = self._python_type_to_json_type(param_type)

            # Starts the schema definition for this parameter
            property_schema = {"type": json_type}

            # Add enum constraint for Literal types
            # Checks whether the annotation restrictions allowed values
            enum_values = self._extract_enum_values(param_type)
            # Adds constraint when allowed values were found
            if enum_values:
                # Lists the permitted values in the parameter schema
                property_schema["enum"] = enum_values

            # Add description from annotation if available using that docstring
            if hasattr(param, "annotation") and hasattr(param.annotation, "__doc__"):
                property_schema["description"] = param.annotation.__doc__

            # Adds the parameter definition to the object schema
            schema["properties"][param_name] = property_schema

            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)

        # Returns the completed JSON schema
        return schema

    def _extract_enum_values(self, param_type: Any) -> Optional[List[Any]]:
        """
        Extract enum values from Literal types or Enum classes.
        Args:
        - param_type: Python type annotation
        Returns:
        - List of enum values or None
        """
        # Handle typing.Literal (Python 3.8+)
        # Checks whether the annotation is a parameterized typing construct
        if hasattr(param_type, "__origin__"):
            # Retrieves its underlying type construct
            origin = getattr(param_type, "__origin__", None)
            # Check for Literal type
            # Attempts tom import helpers for inspecting Literal annotations
            try:
                from typing import Literal, get_args, get_origin

                # Checks whether the annotation is a Literal
                if get_origin(param_type) is Literal:
                    # Returns the values allowed by the Literal
                    return list(get_args(param_type))
            # Handles unavailable typing helpers
            except (ImportError, AttributeError):
                # Continues to the fallback handling
                pass

            # Fallback for older Python or typing_extensions
            # Checks whether the annotation has argument values
            if hasattr(param_type, "__args__"):
                # Check if this looks like a Literal type
                # Creates text to identify the annotation kind
                type_name = str(origin) if origin else str(param_type)
                # Checks whether the annotation appears to be a Literal
                if "Literal" in type_name:
                    # Returns the possible Literal values
                    return list(param_type.__args__)

        # Attempts to extract values from an Enum subclass
        try:
            from enum import Enum

            # Checks whether the annotation is an Enum subclass
            if isinstance(param_type, type_ and issubclass(param_type, Enum):
                # Returns the value of each Enum member
                return [e.value for e in param_type]
        # Handles annotations that cannot be checked this way
        except (TypeError, AttributeError):
            # Treats those annotations as having no extracted enum values
            pass

        # Indicates the no allowed-value list was found
        return None

    # Defines serialization behavior for this function-backed tool
    def dump_component(self):
        """Raise error - FunctionTool cannot be serialized for security reasons."""
        # Rejects attempts to serialize the tool
        raise NotImplementedError(
            f"FunctionTool '{self.name}' cannot be serialized for security reasons. "
            "Consider creating a custom BaseTool subclass for serializable tools."
        )
    
    def _python_type_to_json_type(self, python_type: Any) -> str:
        """
        Convert Python type hints to JSON schema types.
        Args:
        - python_type: Python type annotation
        Returns:
        - JSON schema type string
        """
        # Checks for a missing annotation or the None type
        if python_type is None or python_type == type(None):
            return "null"
        elif python_type == str:
            return "string"
        elif python_type == int:
            return "integer"
        elif python_type == float:
            return "number"
        elif python_type == bool:
            return "boolean"
        # Checks for a plain list or a parameterized list annotation
        elif python_type == list or (
            # Checks whether the annotation's origin is list
            hasattr(python_type, "__origin__") and python_type.__origin__ == list
        ):
            return "array"
        # Checks for a plain dictionary or parameterized dictionary annotations
        elif python_type == dict or (
            # Checks whether the annotation's origin is dict
            hasattr(python_type, "__origin__") and python_type.__origin__ == dict
        ):
            return "object"
        else:
            # For Union types, complex types, etc., default to string
            return "string"
