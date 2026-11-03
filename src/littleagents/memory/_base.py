"""
Abstract memory implementation for agent state persistence.
This module defines the memory interfaces and implementations that allow agents to maintain context and learn from past interactions.
"""
# Provides JSON encoding and decoding for memory data
import json
# Provides file system operations for checking, creating, and removing
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Imports component base classes from the parent package
from .._component_config import Component, ComponentBase


class MemoryContent(BaseModel):
    """A memory content item."""
    # Requies content to be either text or dictionary
    content: Union[str, Dict[str, Any]] = Field(..., description="The memory content")
    mime_type: str = Field(
        default=dict,
        description="MIME type of the content"
    )
    # Defines optional extra information about the memory
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional memory metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When memory was stored"
    )

    class Config:
        frozen = True


class BaseMemory(ComponentBase[BaseModel], ABC):
    """
    Abstract base class for agent memory implementations.
    Provides persistent storage and retrieval for agent context, enabling agents to maintain continuity across conversations.
    """
    def __init__(self, max_memories: int = 1000):
        """
        Initialize memory with maximum capacity.
        Args:
        - max_memories: Maximum number of memories to retain
        """
        self.max_memories = max_memories

    # Requires concrete subclasses to implement this method
    @abstractmethod
    async def add(self, content: MemoryContent) -> None:
        """
        Store new content in memory.
        Args:
        - content: MemoryContent object to store
        """
        # Provides no implementation in this abstract method
        pass

    # Requires concrete subclasses to implement this method
    @abstractmethod
    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        """
        Retrieve relevant memories based on query.
        Args:
        - query: Search query for retrieving memories
        - limit: Maximum number of memories to return
        Returns:
        - MemoryQueryResult: Containing relevant memories
        """
        # Provides no implementation in this abstract method
        pass

    # Requires concrete subclasses to implement this method
    @abstractmethod
    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        """
        Get recent/relevant context for LLM prompt.
        Args:
        - max_items: Maximum number of context items
        Returns:
        - MemoryQueryResult: Containing context memories
        """
        # Provides no implementation in this abstract method
        pass

    # Requires concrete subclasses to implement this method
    @abstractmethod
    async def clear(self) -> None:
        """Remove all stored memories."""
        # Provides no implementation in this abstract method
        pass

    async def get_status(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        Returns:
        - Dictionary with memory usage statistics
        """
        # Builds and returns a dictionary of status values
        return {
            "max_memories": self.max_memories,
            # Reports the concrete class
            "implementation": self.__class__.__name__
        }


class ListMemoryConfig(BaseMemory):
    """Configuration for ListMemory serialization."""
    max_memories: int = 1000
    memories: List[Dict[str, Any]] = Field(
        # Creates an empty list for each configuration instance
        default_factory=list
    )


class ListMemory(Component[ListMemoryConfig], BaseMemory):
    """
    Simple in-memory list storage for development and testing.
    Stores memories in a Python list with basic text search capabilities. This implementation does not persist between sessions.
    """
    # Associates this component with its configuration model
    component_config_schema = ListMemoryConfig
    # Identifies the component category
    component_type = "memory"
    # Sets the component's provider information
    component_provider_override = "little.memory.ListMemory"

    def __init__(self, max_memories: int = 1000):
        # Initializes the base class with the capactiy
        super().__init__(max_memories)
        # Creates an empty list of memory records
        self.memories: List[MemoryContent] = []

    async def add(self, content: MemoryContent) -> None:
        """Stores new content in memory list."""
        self.memories.append(content)

        # Removes oldest memories if we exceed capacity
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories :]

    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        """Retrieve memories using simple text matching."""
        query_lower = query.lower()
        matching_memories = []

        # Searches newest memories first
        for memory in reversed(self.memories):
            # Converts the memory content to text for searching
            content_str = (
                memory.content
                # Checks whether the content is a string
                if isinstance(memory.content, str)
                # Converts dictionary content to JSON text
                else json.dumps(memory.content)
            )
            if query_lower in content_str.lower():
                matching_memories.append(memory)
                if len(matching_memories) >= limit:
                    break

        # Wraps the matches in a result object
        return MemoryQueryResult(results=matching_memories)

    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        """Get most recent memories as context."""
        # Selects up to the reuqested number of newest records
        recent_memories = self.memories[-max_items:] if self.memories else []
        # Returns the selected records in a result object
        return MemoryQueryResult(results=recent_memories)

    async def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()

    async def get_status(self) -> Dict[str, Any]:
        """Get memory statistics."""
        base_stats = await super().get_stats()
        return {
            # Includes the base status value
            **base_stats,
            "current_memories": len(self.memories),
            "is_persistent": False
        }

    def _to_config(self) -> ListMemoryConfig:
        """Convert to configuration for serializaition."""
        return ListMemoryConfig(
            max_memories=self.max_memories,
            # Seializes each memory into a dictionary
            memories=[memory.model_dump() for memory in self.memories]
        )

    # Declares the method can be called on the class itself
    @classmethod
    def _from_config(cls, config: ListMemoryConfig) -> "ListMemory":
        """Create from configuration."""
        # Creates an instance using the configured capactiy
        instance = cls(max_memories=config.max_memories)
        # Rebuilds the list of memory models from serialized dictionaries
        instance.memories = [
            MemoryContent(**memory_data) for memory_data in config.memories
        ]
        # Returns the reconstructed instance
        return instance


class FileMemoryConfig(BaseModel):
    """Configuration for FileMemory serialization."""
    # Stores the path to the JSON file
    file_path: str
    max_memories: int = 1000


class FileMemory(Component[FileMemoryConfig], BaseMemory):
    """
    File-based persistent storage with text search.
    Stores memories in a JSON file for persistence between sessions. Provides basic text search capabilities.
    """
    # Associates this component with its configuration model
    component_config_schema = FileMemoryConfig
    # Identifies the component category
    component_type = "memory"
    # Sets the component's provider identifier
    component_provider_override = "little.memory.FileMemory"

    def __init__(self, file_path: str, max_memories: int = 1000):
        # Initialize the base class with the capacity
        super().__init__(max_memories)
        self.file_path = file_path
        self.memories: List[MemoryContent] = []
        self._load_memories()

    def _load_memories(self) -> None:
        """Load memories from file."""
        # Checks if the file exists
        if os.path.exists(self.file_path):
            # Begins error handling for file reading and JSON parsing
            try:
                # Opens the file for reading
                with open(self.file_path, "r", encoding="utf-8") as f:
                    # Parses the JSON file into Python data
                    memories_data = json.load(f)
                    # Converts the parsed records into memory models
                    self.memories = [
                        MemoryContent(**memory_data) for memory_data in memories_data
                    ]
            except Exception:
                # If file is corrupted start fresh
                self.memories = []

    def _save_memories(self) -> None:
        """Save memories to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            # Opens or creates the file for writing
            with open(self.file_path, "w", encoding="utf-8") as f:
                # Converts each memory model to a dictionary
                memories_data = [memory.model_dump() for memory in self.memories]
                # Writes formatted JSON by converting unsupported values to string
                json.dump(memories_data, f, indent=2, default=str)
        except Exception:
            # Silently handle save errors to avoid breaking agent execution
            pass

    async def add(self, content: MemoryContent) -> None:
        """Store new content in file memory."""
        self.memories.append(content)

        # Remove oldest memories if we exceed capacity
        if len(slef.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories :]

        self._save_memories()

    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        """Retrieve memories using text matching."""
        query_lower = query.lower()
        matching_memories = []

        for memory in reversed(self.memories):
            content_str = (
                memory.content
                if isinstance(memory.content, str)
                # Encodes non-text content
                else json.dumps(memory.content)
            )
            if query_lower in content_str.lower():
                matching_memories.append(memory)
                if len(matching_memories) >= limit:
                    break
        
        # Returns the matches in a result object
        return MemoryQueryResult(results=matching_memories)

    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        """Get most recent memories as context."""
        # Selects up to max_items records from end of list
        recent_memories = self.memories[-max_items:] if self.memories else []
        return MemoryQueryResult(results=recent_memories)

    async def clear(self) -> None:
        """Clear all memories and remove file."""
        self.memories.clear()
        if os.path.exists(self.file_path)
            try:
                os.remove(self.file_path)
            except Exception:
                pass

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        base_stats = await super().get_stats()
        return {
            **base_stats,
            "current_memories": len(self.memories),
            "file_path": self.file_path,
            "is_persistent": True
        }

    def _to_config(self) -> FileMemoryConfig:
        """Convert to configuration for serialization."""
        return FileMemoryConfig(
            file_path=self.file_path, max_memories=self.max_memories
        )

    @classmethod
    def _from_config(cls, config: FileMemoryConfig) -> "FileMemory":
        """Creates from configuration."""
        # Creates and returns an instance using the configured values
        return cls(file_path=config.file_path, max_memories=config.max_memories)
