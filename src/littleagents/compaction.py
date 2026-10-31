"""
Context compaction strategies for LittleAgents.

This module provides strategies for managing context/message history during long-running agent sessions. Strategies are called BEFORE each LLM call in the tool loop, and the returned (potentially compacted) message list continues to the next iteration.
The key insight is that compaction must happen INSIDE the tool loop with reassignment: `messages = strategy.compact(messages)`. This ensures the compacted list is used for subsequent iterations, actually reducing cumulative token usage.
"""
# Imports the decorator used to mark abstract methods
from abc import abstractmethod
# Imports tools for defining data classes and their fields
from dataclasses import dataclass, field
# Imports type hints and runtime-checkable protocol decorator
from typing import List, Optional, Protocol, Set, Tuple, runtime_checkable

from .messages import AssistantMessage, Message, ToolMessage

# Allows runtime checks to test whether an object conforms to this protocol
@runtime_checkable
class CompactionStrategy(Protocol):
    """
    Protocol for context compaction strategies.
    Called BEFORE each LLM call in the tool loop, allowing the strategy to compact the message list. The returned list REPLACES the working message list for subsequent iterations.
    Important: Implementations must preserve "atomic groups" - assistant messages with tool_calls must stay together with their corresponding ToolMessage results. Splitting these causes API errors.
    """
    # Marks this message as required for implementations
    @abstractmethod
    def compact(self, messages: List[Message]) -> List[Messages]:
        """
        Compact messages for the next LLM call.
        Args:
        - messages: Current message list
        Returns:
        - Messages to use (may be compacted). This list REPLACES the working list for subsequent iterations
        """
        # Indicates that the protocol method has no implementation here
        ...


class NoCompaction:
    """
    Baseline: no compaction, context grows unbounded.
    Use this for benchmarks to see how context grows without management, or for short tasks where context limits won't be hit.
    """
    def compact(self, messages: List[Message]) -> List[Message]:
        """Returns messages unchanged."""
        return messages


# Automatically generates common data-class methods for this class
@dataclass
class HeadTailCompaction:
    """
    Token-aware head+tail compaction strategy.
    Preserves:
    - Head: System prompt, initial user message
    - Tail: Recent tool calls and results (working memory)
    Drops middle messages when over budget, respecting atomic groups (tool calls and their results must stay together).
    Args:
    - token_budget: Maximum tokens for context (default: 100,000)
    - head_ratio: Fraction of budget for head messages (default: 20%)
    - model: Model name for tiktoken encoding (default: "gpt-4o")
    """
    # Maximum token budget
    token_budget: int = 100_000
    # Sets the fraction of the budget reserved for the head
    head_ratio: float = 0.2
    # Default model name for tokenizer selection
    model: str = "gpt-4o"

    # Statistics (not included in repr)
    # Introduces internal counters omitted from the generated representation
    # Counts how many times compaction has run
    compaction_count: int = field(default=0, repr=False)
    # Tracks the estimated number of tokens removed
    total_tokens_saved: int = field(default=0, repr=False)

    # Encoder cached on first use
    # Stores the tokenizer (or None if unavailable)
    _encoder: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize tiktoken encoder."""
        try:
            import tiktoken
            # Tries to find an encoding associated with the configuration model
            try:
                # Selects the model-specific tokenizer
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback for uknown models
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            # tiktoken not installed - will use character estimation
            self._encoder = None

    def _count_tokens(self, messages: List[Message]) -> int:
        """
        Count tokens in messages.
        Uses tiktoken if available, falls back to charater estimation.
        """
        # Checks whether a tokenizer is unavailable
        if self._encoder is None:
            # Rough estimate: ~4 chars per token + overhead
            # Estimates tokens from each message's content plus overhead
            return sum(len(str(getattr(m, "content", "") or "")) // 4 + 10 for m in messages)

        total = 0
        # Iterates over messages
        for msg in messages:
            # Role overhead (approximately 4 tokens per message)
            total += 4

            # Retrieves the message content if it exists
            content = getattr(msg, "content", None)
            if content:
                # Adds the encoded content's token count
                total += len(self._encoder.encode(content))

            # Count tool calls if present
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    # Tool call overhead
                    total += 4
                    total += len(self._encoder.encode(tc.tool_name))
                    # Parameters as JSON string
                    import json
                    
                    # Converts the parameters to a JSON string
                    params_str = json.dumps(tc.parameters)
                    # Adds tokens used by serializable parameters
                    total += len(self._encoder.encode(params_str))

        # Returns the accumulated token estimate
        return total

    # Finds message-index groups that should not be seperated
    def _find_atomic_groups(self, messages: List[Message]) -> List[Tuple[int, ...]]:
        """
        Group tool_call messages with their results.
        OpenAI/Anthropic require every tool_call to have a corresponding result. This ensures we never split a tool call from its results.
        Returns:
        - List of tuples, where each tuple contains indices that must stay together.
        """
        # Creates a list to hold groups of message indicies
        groups: List[Tuple[int, ...]] = []
        i = 0

        while i < len(messages):
            msg = message[i]

            if isinstance(msg, AssistantMessage) and msg.too_calls:
                # This message has tool calls - find all results
                # Collects the IDs of the assistant's tool calls
                call_ids: Set[str] = {tc.call_id for tc in msg.tool_calls}
                # Starts the group with the assistant message's index
                group_indicies = [i]

                # Look ahead for results
                j = i + 1
                while j < len(messages) and call_ids:
                    # Checks if the current message is a tool result
                    if isinstance(messages[j], ToolMessage):
                        tool_msg = messages[j]
                        # Checks whether the result matches a call in this group
                        if tool_msg.tool_call_id in call_ids:
                            group_indicies.append(j)
                            call_ids.remove(tool_msg.tool_call_id)

                    j += 1

                # Adds the tool-call group to the list of groups
                groups.append(tuple(group_indicies))
                # Continues after the last grouped message, or advances one position
                i = max(group_indicies) + 1 if group_indicies else i + 1
            else:
                # Adds the message as a group of one
                groups.append((i,))
                i += 1

        # Returns all the groups of message indicies
        return groups

    def compact(self, messages: List[Message]) -> List[Message]:
        """
        Compact message if over budget.
        Preserves head (system prompt, initial context) and tail (recent work), dropping middle messages when necessary.
        """
        # Checks if the input list is empty
        if not messages:
            return messages

        # Counts tokens before compaction
        current_tokens = self._count_tokens(messages)

        # Checks whether the list already fits within budget
        if current_tokens <= self.token_budget:
            return messages

        # Compaction needed
        self.compaction_count += 1

        # Groups messages that must remain together
        groups = self._find_atomic_groups(messages)
        # Calculates the token allowance for the head
        head_budget = int(self.token_budget * self.head_ratio)
        # Assigns the remaining budget to the tail
        tail_budget = self.token_budget - head_budget

        # Fill head from start
        head_groups: List[Tuple[int, ...]] = []
        head_tokens = 0

        # Iterates over message groups from the beginning
        for group in groups:
            # Retrieves the messages in the current group
            group_msgs = [messages[i] for i in group]
            # Counts tokens in the current group
            group_tokens = self._count_tokens(group_msgs)

            # Checks whether this group fits in the head budget
            if head_tokens + group_tokens <= head_budget:
                head_groups.append(group)
                head_tokens += group_tokens
            # Runs when the next group would would exceed the head budget
            else:
                # Stops adding groups to the head
                break

        # Fill tail from end (skip head groups)
        # Gets the groups after the head
        remaining_groups = groups[len(head_groups) :]
        # Creates a list for selected tail groups
        tail_groups: List[Tuple[int, ...]] = []
        # Initializes the tail token count
        tail_tokens = self._count_tokens(group_msgs)

        # Checks whether a group fits in the tail budget
        if tail_tokens + group_tokens <= tail_budget:
            # Adds the group to the beginning of the tail list
            tail_groups.insert(0, group)
            tail_tokens += group_tokens
        # Runs when the group would exceed the tail budget
        else:
            # Stops adding groups
            break

        # Build compacted list
        # Creates a set of retained indicies
        kept_indicies = Set[int] = set()
        # Iterates over groups selected for preservation
        for group in head_groups + tail_groups:
            # Adds all indiices from the group to the retained-index set
            kept_indicies.update(group)

        # Keeps selected messages in their original order
        compacted = [messages[i] for i in sorted(kept_indicies)]

        # Track savings
        # Counts tokens in the compacted list
        compacted_tokens = self._count_tokens(compacted)
        # Adds the number of removed tokens to the running total
        self.total_tokens_saved += current_tokens - compacted_tokens
        
        # Returns the compacted message list
        return compacted


# Automatically generates common data-class methods for this class
@dataclass
class SlidingWindowCompaction:
    """
    Keep only recent messages within budget.
    Always preserves the system message (if present) plus the most recent messages that fit in the budget. Respects atomic groups (tool calls and their results must stay together.
    Simpler than HeadTailCompaction but may lose important early context. Best for conversational agents where recent context matters most.
    Args:
    - token_budget: Maximum tokens for context (default: 100,000)
    - model: Model name for tiktoken encoding (default: "gpt-4o")
    """
    # Sets the maximum token budget
    token_budget: int = 100_000
    # Sets the default model name for tokenizer selection
    model: str = "gpt-4o"

    # Statistics
    # Counts how many times compaction has run
    compaction_count: int = field(default=0, repr=False)
    # Tracks the estimated number of tokens removed
    total_tokens_saved: int = field(default=0, repr=False)

    # Encoder cached on first use
    # Stores the tokenizer or None if unavailable
    _encoder: Optional[object] = field(default=None, repr=False)

    # Runs after initialization to set up the tokenizer
    def __post_init__(self) -> None:
        """Initialize tiktoken encoder."""
        try:
            import tiktoken

            # Tries to find an encoding associated with the configuration model
            try:
                # Selects the model-specific tokenizer
                self._encoder = tiktoken.encoding_for_model(self.model)
            # Handles the case where tiktokens is not installed
            except KeyError:
                # Selects a fallback tokenizer
                self._encoder = tiktoken.get_encoding("cl100k_base")
        # Handles the cases where tiktoken is not installed
        except ImportError:
            # Signals that the token counts should be estimated from text length
            self._encoder = None

    def _count_tokens(self, messages: List[Messages]) -> int:
        """Count tokens in message."""
        # Checks if a tokenizer is unavailable
        if self._encoder is None:
            # Estimates tokens from each message's content plus overhead
            return sum(len(str(getattr(m, "content", "") or "")) // 4 + 10 for m in messages)

        total = 0
        #  Iterates over each message
        for msg in messages:
            total += 4
            # Retrieves the message content if it exists
            content = getattr(msg, "content", None)
            if content:
                # Adds the encoded content's token count
                total += len(self._encoder.encode(content))
            # Checks whether this is an assistant message with tool calls
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                # Iterates over each tool call
                for tc in msg.tool_calls:
                    # Adds estimated overhead and the tool name's token
                    total += 4 + len(self._encoder.encode(tc.tool_name))

                    import json

                    # Counts parameter tokens
                    total += len(self._encoder.encode(json.dumps(tc.parameters)))

        # Returns the accumulated token estimate
        return total

    # Finds message-index groups that should not be seperated
    def _find_atomic_groups(self, messages: List[Message]) -> List[Tuple[int, ...]]:
        """Group tool_call messages with their results."""
        # Creates a list to hold groups of message indicies
        groups: List[Tuple[int, ...]] = []
        i = 0

        # Continues until every message is considered
        while i < len(messages):
            msg = messages[i]
            
            # Checks for an assistant message with tool calls
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                # Collects the IDs of the assistant's tool calls
                call_ids: Set[str] = {tc.call_id for tc in msg.tool_calls}
                # Starts the group with the assistant message's index
                group_indicies = [i]

                j = i + 1
                # Searches until the end or until all matching results are found
                while j < len(messages) and call_ids:
                    # Checks whether the current message is a tool result
                    if isinstance(messages[j], ToolMessage):
                        # Stores the tool result
                        tool_msg = messages[j]
                        # Checks whether the reult matches a call in this group
                        if tool_msg.tool_call_id in call_ids:
                            # Adds the result's index to the group
                            group_indicies.append(j)
                            # Marks this tool call as matched
                            call_ids.remove(tool_msg.tool_call_id)

                    j += 1

                # Adds the tool-call group to the list of groups
                groups.append(tuple(group_indicies))
                # Continues after the last group message or advances one posiiton
                i = max(group_indicies) + 1 if group_indicies else i + 1
            # Handles messages that do not begin a tool-call group
            else:
                # Adds the message as a group of one
                groups.append((i,))
                i += 1

        # Returns all the groups of the message indicies
        return groups

    def compact(self, messages: List[Message]) -> List[Message]:
        """Keep system message + most recent messages within budget."""
        # Checks whether the input list is empty
        if not messages:
            return messages

        # Counts tokens before compaction
        current_tokens = self._count_tokens(messages)

        # Compares the count with token budget
        if current_tokens <= self.token_budget(messages):
            return messages

        # Compaction needed
        self.compaction_count += 1

        # Groups messages that must remain together
        groups = self._find_atomic_groups(messages)

        # Always keep system message if present (first message, first group)
        # # Creates a list for groups containing system messages
        system_groups: List[Tuple[int, ...]] = []
        system_tokens = 0

        # Checks whether the first message is a system message
        if groups and messages[groups[0][0]].role == "system":
            # Preserves the first group
            system_groups.append(groups[0])
            # Counts tokens in that group
            system_tokens = self._count_tokens([messages[i] for i in groups[0]])
            # Removes the preserved system group from the remaining candidates
            groups = groups[1:]

        # Fill from end with remaining budget
        # Calculates the budget available after preserving the system message
        remaining_budget = self.token_budget - system_tokens
        # Creates a list for the recent groups that fit
        kept_groups: List[Tuple[int, ...]] = []
        kept_tokens = 0

        # Iterates over remaining groups from newest to oldest
        for group in reversed(groups):
            # Retrieves the messages in the current group
            group_msgs = [messages[i] for i in group]
            # Counts tokens in the current group
            group_tokens = self._count_tokens(group_msgs)

            # Checks whether the group fits within the remaining budget
            if kept_tokens + group_tokens <= remaining_budget:
                # Adds the group to `kept_groups`
                kept_tokens.insert(0, group)
                kept_tokens += group_tokens
            # Runs when the group would exceed the remaining budget
            else:
                # Stops adding older groups once the next one does not fit
                break

        # Build compacted list
        # Creates a set of retained indicies
        kept_indicies: Set[int] = set()
        # Iterates over preserved system and recent groups
        for group in system_groups + kept_groups:
            # Adds all indicies from the group to the retained-index set
            kept_indicies.update(group)

        # Keeps selected messages in their original order
        compacted = [messages[i] for i in sorted(kept_indicies)]

        # Track savings
        # Counts tokens in the compacted list
        compacted_tokens = self._count_tokens(compacted)
        # Adds the number of removed tokens to the running total
        self.total_tokens_saved += current_tokens - compacted_tokens

        # Returns the compacted message list
        return compacted
