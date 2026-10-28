from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseChatCompletionClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> 'ChatCompletionResult':
        """Make a single API call."""
        pass

    @abstractmethod
    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator['ChatCompletionChunk', None]:
        """Make a streaming LLM API call."""
        pass

class OpenAIChatCompletionClient(BaseChatCompletionClient):
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None
    ) -> ChatCompletionResult:
        # Step 1: Convert our types to provider's format
        api_messages = self._convert_messages_to_api_format(messages)

        # Step 2: Make the provider specific API call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=tools
        )

        # Step 3: Convert response back to our unified format
        return ChatCompletionResult(
            message=AssistantMessage(
                content=response.choices[0].message.content
            ),
            usage=Usage(
                token_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens
            ),
            model=response.model
        )

class Agent(BaseAgent):
    async def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]]
    ) -> AsyncGenerator[Union[Message, AgentEvent], None]:
        # Prepare context with instructions and history
        llm_messages = [
            SystemMessage(content=slef.instructions),
            *self.context.messages,
            *task_messages
        ]

        # Call model client
        completion_result = await self.model_client.create(llm_messages)
        assistant_message = completion_result.message

        # Yield the response
        yield assistant_message

        # Update conversation context
        self.context.add_message(assistant_message)
