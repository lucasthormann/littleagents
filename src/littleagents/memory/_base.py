from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseMemory(ABC):
    """Abstract interface for agent memory systems."""

    @abstractmethod
    async def add(
        self,
        content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store new content in memory."""
        pass

    @abstractmethod
    async def query(self, query: str, limit: int = 10) -> List[str]:
        """Retrieve relevant memories based on query."""
        pass

    @abstractmethod
    async def get_context(self, max_items: int = 10) -> List[str]:
        """Get recent/relevant context for LLM prompt."""
        pass

    async def _prepare_llm_messages(
        self,
        task_messages: List[Message]
    ) -> List[Message]:
        """Prepare context with memory integrations."""
        system_content = self.instructions

        # Memory provides relevant context through get_context()
        if self.memory:
            context = await self.memory.get_context(max_items=5)
            if context:
                system_content += f"\n\nRelevant context:\n{'\n'.join(context)}"

        return [
            SystemMessage(content=system_content),
            *self.message_history,
            *task_messages
        ]

class MemoryBackend:
    def _validate_path(self, path: str) -> Path:
        """Prevent directory traversal attack."""
        # Resolve to absolute path
        full_path = (self.base_path / path).resolve()

        # Ensure path stays within base directory
        try:
            full_path.relative_to(self.base_path)
        except ValueError:
            raise ValueError(
                "Access denied: path outside memory"
            )

        return full_path
