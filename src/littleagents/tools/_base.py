# Structured output enables reliable tool requests
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

# Agent produces this structured reliably
result = await model_client.create( 
    messages=messages,
    tools=available_tools, 
    # Tool schemas guide structured output
)

# Now we can safely execute the tool
if result.tool_calls:
    for tool_call in tool_calls:
        tool_result = await execute_tool(tool_call.name, tool_call.arguments)

class BaseTool(ABC): 
    """Abstract base class for all tools."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON schema defining expected inputs."""
        pass

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def to_llm_format(self) -> Dict[str, Any]:
        """Convert to OpenAI (or any other) function calling format."""

"""
Converting mixed tool types to standard format.
"""
class Agent(BaseAgent):
    def _process_tools(
        self,
        tools: List[Union[BaseTool, Callable]]
    ) -> List[BaseTool]:
        """Convert mixed tool types to BaseTool instances."""
        processed = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                processed.append(tool)
            elif callable(tool):
                # Wrap functions in FunctionTool automatically
                processed.append(FunctionTool(tool))
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        return processed

    def _get_tools_for_llms(self) -> List[Dict[str, Any]]:
        """Convert tools to LLM function calling format."""
        return [tool.to_llm_format() for tool in self.tools]

    
    async def _exectue_tool_call(
        self,
        tool_call: ToolCallRequest
    ) -> ToolMessage:
        """Execute a tool call and return result message."""
        # Find the requested tool by name
        tool = self._find_tool(tool_call.tool_name)
        if tool is None:
            return ToolMessage(
                content=f"Tool '{tool_call.tool_name}' not found",
                tool_call_id=tool_call.call_id,
                success=False,
                error=f"Tool not found"
            )

        # Execute the tool with error handling
        try:
            result = await tool.execute(tool_call.parameters)
            content = (
                str(result.result) if result.success
                else f"Error: {result.error}"
            )
            return ToolMessage(
                content=content,
                tool_call_id=tool_call.tool_name,
                success=result.success,
                error=result.error
            )
        except Exception as e:
            # Handle unexpected errors gracefully
            return ToolMessage(
                content=f"Tool execution failed: {str(e)}",
                tool_call_id=tool_call.call_id,
                success=False,
                error=str(e)
            )



