# Agent maintains conversation context
class AgentContext:
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, message: Message):
        self.messages.append(message)
        # Optionally truncate to keep recent context
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

# Usage in agent execution loop
self.context.add_message(
    UserMessage(content="What is the height of the Eiffel Tower?"))
self.context.add_message(
    AssistantMessage(content="The Eiffel Tower is 324 meters tall."))
self.context.add_message(
    UserMessage(content="When was it built?"))

# All messages become part of the prompt to the LLM
llm_messages = [SystemMessage(content=self.instructions), *self.context.messages]
