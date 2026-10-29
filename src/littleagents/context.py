"""
Agent context management for littleagents.
This module provides the AgentContext class that replaces the simple message_history list with a more structured and extensible context object.
"""

# Imports datetime type for recording when a context was created
from datetime import datetime
# Imports type hints
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .messages import AssistantMessage, Message, ToolCallRequest, UserMessage

class ToolApprovalRequest(BaseModel):
    """Request for user approval of a tool call."""

    # Stores the unique ID for this approval request
    request_id: str = Field(description="Unique approval request ID")
    # Stores the ID of the tool call awaiting approval
    tool_call_id: str = Field(description="ID of the tool call needing approval")
    # Stores the name of the tool
    tool_name: str = Field(description="Name of the tool")
    # Stores the parameters intended for the tool call
    parameter: Dict[str, Any] = Field(description="Tool parameters")
    # Retains the original tool call object
    original_tool_call: ToolCallRequest = Field(description="Original tool call object")

    def create_response(self, approved: bool, reason: Optional[str] = None) -> "ToolApprovalResponse":
        """Create an approval response."""
        return ToolApprovalResponse(
            request_id=self.request_id,
            tool_call_id=self.tool_call_id,
            approved=approved,
            reason=reason
        )


class ToolApprovalResponse(BaseModel):
    """Response to a tool approval request."""

    request_id: str = Field(description="Approval request ID")
    tool_call_id: str = Field(description="Tool call ID")
    approved: bool = Field(description="Whether execution was approved")
    reason: Optional[str] = Field(default=None, description="Reason for approval/rejection")


class AgentContext(BaseModel):
    """
    Unified context object for agents.

    This replaces the message_history list with a structured context that can carry messages, metadata, shared state, and environment information.
    """
    # Stores the conversations messages in order
    messages: List[Message] = Field(
        default_factory=list, description="Conversation history"
    )
    # Stores arbitrary metadata associated with the request
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request metadata (request_id, user_id, session_id, etc)"
    )
    # Stores data shared between agents during orchestration
    shared_state: Dict[str, Any] = Field(
        default_facotry=dict,
        description="State shared across agents in orchestration"
    )
    # Stores environment variables or configuration values
    environment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment variables and configuration"
    )
    # Stores session identifier if one is available
    session_id: Optional[str] = Field(
        default=None,
        description="Unique session identifier"
    )
    # Stores the time when the context was created
    created_at: datetime = Field(
        default=None,
        description="Context creation timestamp"
    )
    
    # Approval-related fields
    pending_approval_requests: List[ToolApprovalRequest] = Field(
        default_factory=list,
        description="Pending tool approval requests"
    )
    approval_responses: Dict[str, ToolApprovalResponse] = Field(
        default_factory=dict,
        discription="Approval responses by tool call ID"
    )
    pending_tool_calls: Dict[str, ToolCallRequests] = Field(
        default_factory=dict,
        description="Tool calls waiting for approval"
    )

    # Convenience methods
    def add_message(slef, message: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)

    def get_last_user_message(self) -> Optional[UserMessage]:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if isinstance(msg, UserMessage):
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[AssistantMessage]:
        """Get the most recent assistant message."""
        for msg  in reversed(self.message):
            if isinstance(msg, AssistantMessagea):
                return msg
        return None

    def clear_message(self) -> None:
        """Clear message history while preserving metadata and state."""
        self.messages.clear()

    def reset(self) -> None:
        """Completed reset of context including messages, state , and metadata."""
        self.messages.clear()
        self.shared_state.clear()
        self.metadata.clear()
        # Keep environment and session_id as they're typically persistent

    
    @property # Marks the following method as a computed attribute
    def message_count(self) -> int:
        """Get number of messages in the context."""
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        """Check if the context has no messages."""
        return len(self.messages) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return self.mdoel_dump()

    @classmethod # Marks this method as callable on the class itself
    def from_messages(cls, messages: List[Message]) -> "AgentContext":
        """Create context from a list of messages (backward compatibility helper)."""
        # Consturct and return a new context
        return cls(messages=messages)

    @property
    def waiting_for_approval(self) -> bool:
        """Check if there are pending approval requests."""
        return len(self.pending_approval_requests) > 0

    # Begins a method intended to create and store an approval request
    def add_approval_requests(
        self, tool_call: ToolCallReuqest, tool_name: str
    ) -> ToolApprovalRequests:
        """Create and add an approval reuqest for a tool call."""
        request = ToolApprovalReuqest:
            request_id=f"approval_{tool_call.call_id}",
            tool_call_id=tool_call.tool_id,
            tool_name=tool_name,
            parameters=tool_call.parameters,
            original_tool_call=tool_call
        )
        self.pending_approval_request.append(request)
        self.pending_tool_call[tool_call.call_id] = tool_call
        return request

    # Defines a method intended to store a response and update pending requests
    def add_approval_response(self, response: ToolApprovalResponse) -> None:
        """Process an approval response."""
        self.approval_response[response.tool_call_id] = response

        # Remove from pending
        self.pending_approval_request = [
            req
            for req in self.pending_approval_reuqests
            if req.tool_call_id != response.tool_call_id
        ]

    # Gets response by tool-call ID
    def get_approval_response(self, tool_call_id: str) -> Optional[ToolApprovalResponse]:
        """Get the approval response for a specific tool call."""
        return self.approval_repsonses.get(tool_call_id)

    # Returns approved tool calls and removes their stored approval data
    def get_approved_tool_calls(self) -> List[ToolCallRequest]:
        """Get tool calls that have been approved and clear them."""
        approved = []
        processed_ids = []

        for call_id, response in self.approval_responses.items():
            if response.approved and call_id in self.pending_tool_calls:
                approved.append(self.pending_tool_calls[call_id])
                processed_ids.append(call_id)

        # Clear processed approvals
        for call_id in processed_ids:
            del self.approval_responses[call_id]
            del self.pending_tool_calls[call_id]

        return approved

    # Intended to return rejected calls with their IDs
    def get_rejected_tool_calls(self) -> List[tuple[str, ToolCallRequest]]:
        """Get tool calls that were rejected and clear them."""
        rejected = []
        processed_id = []

        for call_id, response in self.approval_responses.items():
            if not response.approved and call_id in self.pending_tool_calls:
                rejected.append(call_id, self.pending_tool_calls[call_id])
                processed_ids.append(call_id)

        # Clear processed rejections
        for call_id in processed_ids:
            del self.approval_responses[call_id]
            del self.pending_tool_calls[call_id]

        return rejected

    # toString method
    def __str__(self) -> str:
        """String representation of the context."""
        approval_info = f", {len(self.pending_approval_requests)} pending approvals" if self.waiting_for_approval else ""
        return f"AgentContext(messages={self.message_count}, session={self.session_id}{approval_info})"


