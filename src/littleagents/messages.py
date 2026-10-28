"""
Core message types for agent communication using Pydantic models.
This module defines the structured message types that agents use to communicate
with each other and with LLMs, following the OpenAI API format.
"""

from datetime import datetime # Import the datetime type used for message timestamps
from typing import Any, Dict, List, Literal, Optional, Union, TYPE_CHECKING # imoprt type annotations used throughout the models

# Import Pydantic's model base class, field helper, and validation decorator
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from .types import Usage

class BaseMessage(BaseModel):
    """Base class for all message types."""

    content: str = Field(..., description="The message content")
    source: str = Field( # Declares the required message source
        ..., description="Source of the message (agent name, system, user, etc.)"
    )
    timestamp: datetime = Field( # Declares the message creation time
        default_factory=datetime.now, description="When the message was created"
    )

    # Defines Pydantic model configuration
    class Config:
        # Prevents fields from being reassigned after the model is created
        frozen = True

    def __str__(self) -> str:
        """Returns a user-friendly string representation."""
        time_str = self.timestamp.strftime("%H:%M:%S") # Format timestamp
        return f"[{self.source}] {time_str} | {self.content}"

    def __repr__(self) -> str:
        """Returns an unambiguous, developer-friendly represetnation."""
        class_name = self.__class__.__name__ # Gets the concrete model class name
        return f"{class_name}(source='{self.source}', content='{self.content[:50]}...', timestamp='{self.timestamp}')"


class SystemMessage(BaseMessage):
    """System message containing instructions/role definition for the agent."""
    # Fixes this message's role to "system"
    role: Literal["system"] = Field(default="system", description="Message role")

class UserMessage(BaseMessage):
    """User message containing input from human or external system."""
    # Fixes this message's role to "user"
    role: Literal["user"] = Field(default="user", description="Message role")
    # Optionally stores the user's name
    name: Optional[str] = Field(default=None, description="Optional name of the user")


class ToolCallRequest(BaseModel):
    """Structured representation of an LLM's tool call request."""
    # Requires the name of the tool to invoke
    tool_name: str = Field(..., description="Name of the tool to call")
    # Requires a dictionary of tool arguments
    parameters: Dict[str, Any] = Field(..., description="Arguments for the tool")
    # Requires an ID for matching the call with its result
    call_id: str = Field(..., description="Unique identifier for this call")

    class Config:
        frozen = True


class AssistantMessage(BaseMessage):
    """Assistant message containing response from the agent/LLM."""
    # Fixes this message's role to "assistant"
    role: Literal["assistant"] = Field(default="assistant", description="Message role")
    # Declares an optional list of tool calls
    tool_calls: Optional[List[ToolCallRequest]] = Field(
        default=None, description="Tool calls made by the assistant"
    )
    # Declares optional structured output validated as a Pydantic model
    structured_content: Optional[BaseModel] = Field(
        default=None, description="Structured data when output_format is used"
    )
    # Declares optional token-usage information, using a forward reference
    usage: Optional["Usage"] = Field(
        default=None, description="Token usage for this LLM call"
    )

    # Overrides the base string conversion to include tool-call details
    def __str__(self) -> str:
        """Returns a user-friendly string representation."""
        time_str = self.timestamp.strftime("%H:%M:%S")

        # Checks whether the assistant made any tool calls
        if self.tool_calls:
            # Show tool calls information
            tool_info = ", ".join(
                [
                    f"{tc.tool_name}({', '.join(f'{k}={v}' for k, v in tc.parameters.items())})"
                    # Iterates over each requested tool
                    for tc in self.tool_calls
                ]
            )
            # Checks whether the message contains non-whitespace text
            if self.content and self.content.strip():
                return (
                    f"[{self.source}] {time_str} | {self.content} [tools: {tool_info}]"
                )
            else:
                return f"[{self.source}] {time_str} | [calling tools: {tool_info}]"
        else:
            return f"[{self.source}] {time_str} | {self.content}"


class ToolMessage(BaseMessage):
    """Tool message containing result from tool execution."""
    # Fixes this message's role to "tool"
    role: Literal["tool"] = Field(default="tool", description="Message role")
    # Declares the ID of the tool call that this result corresponds to
    tool_call_id: str = Field(
        ..., description="ID of the tool call this is responding to"
    )
    # Requires the name of the executed tool
    tool_name: str = Field(..., description-"Name of the tool that was executed")
    # Requires a flag indicating whether the tool succeeded
    success: bool = Field(..., description="Whether tool execution succeeded")
    # Optionally stores an error message when execution fails
    error: Optional[str] = Field(default=None, description="Error message if failed")
    # Declares a dictionary for additional tool-specific information
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Tool-specific metadata (e.g. sub-agent usage)"
    )


class MultiModalMessage(BaseMessage):
    """Message supporting multiple content types (text, images, audio, etc.)."""
    # Requires the role to be either "user" or "assistant"
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    # Declares the media type using MIME-type string
    mime_type: str = Field(
        ...,
        description="MIME type of the content (e.g. 'text/plain', 'image/jpeg', 'audio/wav', 'video/mp4')",
    )
    # Allows content to be supplied as bytes or as a string
    data: Optional[Union[bytes, str]] = Field(
        default=None, description="Binary data (bytes) or base64 string for the contnet"
    )
    # Declares an optional URL pointing to media content
    media_url: Optional[str] = Field(
        default=None, description="URL to media content if data is not provided"
    )
    # Declares a dictionary for additional content information
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional content metadata"
    )


    # Runs the validation after the mode fields have been parsed
    @model_validator(mode="after")
    def validate_media_data(self):
        """Ensure either data or media_url is provided."""
        # Checks whether both content sources are missing
        if self.data is None and self.media_url is None:
            raise ValueError("Either 'data' or 'media_url' must be provided")
        # Checks whether both content sources were supplied
        if self.data is not None and self.media_url is not None:
            raise ValueError("Only one of 'data' or 'media_url' should be provided")

        # Returns the validated model instance
        return self

    # Helpers that check what format the MIME type indicates
    def is_text(self) -> bool:
        """Check if this is a text message."""
        return self.mime_type.startswith("text/")

    def is_image(self) -> bool:
        """Check if this is an image message."""
        return self.mime_type.startswith("image/")

    def is_audio(self) -> bool:
        """Check if this is an audio message."""
        return self.mime_type.startswith("audio/")

    def is_video(self) -> bool:
        """Check if this is a video message."""
        return self.mime_type.startswith("video/")

    def to_base64(self) -> Optional[str]:
        """Convert data to base64 string for API usage."""
        if self.data is None:
            return None
        
        # If data is already a string, assume it's base64
        if isinstance(self.data, str):
            return self.data
        
        # If data is bytes, encode to base64
        import base64

        # Base64-encodes the bytes and converts the result to text
        return base64.b64encode(self.data).decode("utf-8")


# Union type for all message types
Message = Union[
    SystemMessage, UserMessage, AssistantMessage, ToolMessage, MultiModalMessage
]
