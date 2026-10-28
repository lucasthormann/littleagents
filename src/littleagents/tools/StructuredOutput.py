from pyndatic import BaseModel, Field
from typing import List

class UserProfile(BaseModel):
    id: int = Field(..., description="The user's unique identifier")
    name: str = Field(..., description="The user's full name")
    email: str = Field(..., description="The user's email address")
    age: int = Field(..., description="The user's age")
    skills: List[str] = Field(..., description="A list of the user's skills")

# Structured output enables reliable tool request
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

# Agent produces this structure reliably
result = await model_client.create(
    messages=messages,
    tools=available_tools,
    # Tool schemas guide structured output
)

# Now we can safely execute the tool
if result.tool_calls:
    for tool_call in result.tool_calls:
        tool_result = await execute_tool(tool_call.name, tool_call.arguments)
