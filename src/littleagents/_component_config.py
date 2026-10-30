"""
Component configuration system for LittleAgents.

This module provides a flexible component serialization and loading system that allows components to be saved as JSON configuration and loaded back as instances. This is foundational for workflow persistence, UI integration, and system interoperability.
"""
# Postpones evaluation of type annotations
from __future import annotations

# Provides functions for importing modules dynamically
import importlib
import warnings
from typing import Any, ClassVar, Dict, Generic, Literal, Type, Union, cast, overload

from pydantic import BaseModel
from typing_extensions import Self, TypeGuard, TypeVar

# Defines the accepted type for a component's category
ComponentType = Union[
    Literal[
        "model", "agent", "tool", "termination", "orchestrator", "step", "workflow"
    ],
    str,
]
# Declares a type variable for Pydantic configuration models
ConfigT = TypeVar("CofigT", bound=BaseModel)
# Declares a contravariant config type for loading
FromConfigT = TypeVar("FromConfigT", bound=BaseModel, contravariant=True)
# Declares a covariant config type for dumping
ToConfigT = TypeVar("ToConfigT", bound=BaseModel, covariant=True)

# Declares a covariant type variable bounded by BaseModel
T = TypeVar("T", bound=BaseModel, covariant=True)

class ComponentModel(BaseModel):
    """
    Model class for a serializable component.
    Contains all information required to instantiate a component including:
    - Provider information for class
    - Type and version metadata
    - Configuration data for instantiation
    This enables components to be saved as JSON configurations and loaded back as instances, supporting workflow persistence and UI integration.
    """
    # Stores the component class's provider path or alias
    provider: str

    # Stores the component category
    component_type: ComponentType | None = None

    # Stores a general version number
    version: int | None = None

    # Stores the component configuaration version
    component_version: int | None = None
    
    # Stores a human readable description
    description: str | None = None

    # Stores a display label
    label: str | None = None

    # Stores the component's congfiguration values
    config: dict[str, Any]

# Convert a class into a provider path string
def _type_to_provider_str(t: type) -> str:
    # Combines its module and qualified class name
    return f"{t.__module__}.{t.__qualname__}"

# Maps short aliases to fully qualified component paths
WELL_KNOWN_PROVIDERS = {
    #LittleAgents Model Client
    "openai_chat_completion_client": "little.llm.OpenAIChatCompletionClient",
    "OpenAIChatCompletionClient": "little.llm.OpenAIChatCompletionClient",
    "model_client": "little.llm.OpenAIChatCompletionClient",

    #LittleAgents Agents
    "agent": "little.agents.Agent",
    "Agent": "little.agents.Agent",
    
    # Little Agents AgentsMemory
    "list_memory": "little.memory.ListMemory",
    "ListMemory": "little.memory.ListMemory",
    "file_memory": "little.memory.FileMemory",
    "FileMemory": "little.memory.FileMemory",
    "memory": "little.memory.ListMemory",
    
    # Little Agents AgentsTermination
    "max_message_termination": "little.termination.MaxMessageTermination",
    "MaxMessageTermination": "little.termination.MaxMessageTermination",
    "text_mention_termination": "little.termination.TextMentionTermination",
    "TextMentionTermination": "little.termination.TextMentionTermination",
    "composite_termination": "little.termination.CompositeTermination",
    "CompositeTermination": "little.termination.CompositeTermination",
    "termination": "little.termination.MaxMessageTermination",
    
    # Little Agents AgentsOrchestrators
    "round_robin_orchestrator": "little.orchestration.RoundRobinOrchestrator",
    "RoundRobinOrchestrator": "little.orchestration.RoundRobinOrchestrator",
    "ai_orchestrator": "little.orchestration.AIOrchestrator",
    "AIOrchestrator": "little.orchestration.AIOrchestrator",
    "orchestrator": "little.orchestration.RoundRobinOrchestrator",
    
    # Little Agents Workflow Components
    "workflow": "little.workflow.Workflow",
    "Workflow": "little.workflow.Workflow",
    "littleagent_step": "little.workflow.LittleAgentStep",
    "LittleAgentStep": "little.workflow.LittleAgentStep",
}

# Defines the interface for creating components from configuration
class ComponentFromConfig(Generic[FromConfigT]):
    """
    Mixin class for components that can be loaded from configuration.
    Implement this interface to enable your component to be instantiated from a seializable configuration object.
    """
    # Defines the expected config-loading method
    @classmethod
    def __from_config(cls, config: FromConfigT) -> Self:
        """
        Create a new instance of the component from a configuration.
        Args:
        - config: The validated configuration object (Pydantic model)
        Returns:
        - A new instance of the component initialized from the config
        Raises:
        - NotImplementedError: If the component doesn't support cofig loading
        """
        raise NotImplementedError("This component does not support loading from config")

    # Defines a hook for loading older config versions
    @classmethod
    def _from_config_past_version(cls, config: Dict[str, Any], verison: int) -> Self:
        """
        Create a new instance of the component from a previous version of the configuaration object.
        Args:
        - config (Dict[str, Any]): The configuration object
        - version (int): The version of the configuration object
        Returns:
        - Self: the new instance of the component

        :meta public:
        """
        raise NotImplementedError(
            "This component does not support loading from past versions"
        )

# Defines the interface for components that can be serialized to config
class ComponentToConfig(Generic[ToConfigT]):
    """
    The two methods a class must implement to be a component.
    Args:
    - Protocol (ConfigT): Type which derives from :py:class:`pydantic.BaseModel`
    """
    # Declares the component category as a class-level attribute
    component_type: ClassVar[ComponentType]

    # Declares a default component version
    component_version: ClassVar[int] = 1

    # Allows subclasses to override the provider path
    component_provider_override: ClassVar[str | None] = None

    # Allows subclasses to set a description
    component_description: ClassVar[str | None] = None

    # Allows subclasses to set a display label
    component_label: ClassVar[str | None] = None

    # Defines the method subclasses use to produce configuration data
    def _to_config(self) -> ToConfigT:
        """
        Dump the configuration that would be required to create a new instance of a component matching the configuration of this instance.
        Returns:
        - T: The configuration of the component

        :meta public
        """
        raise NotImplementedError("This component does not support dumping to config")

    # Builds a serializable model representing this component
    def dump_component(self) -> ComponentModel:
        """
        Dump the component to a model that can be loaded back in
        Raises:
        - TypeError: If the component is a local class
        Returns:
        - ComponentModel: The model representing the component
        """
        # Checks if a custom provider path was supplied
        if self.component_provider_override is not None:
            provider = self.component_provider_override
        else:
            provider = _type_to_provider_str(self.__class__)
            if "._" in provider:
                warnings.warn(
                    "Internal module name used in provider string. This is not recommended and may cause issues in the future. Silence this warning by setting component_provider_override to this value.",
                    stacklevel=2
                )
            # Checks if the class is local and not importable by a stable path
            if "<local>" in provider:
                raise TypeError("Cannot dump component with local class")
            
            # Checks that a component category is available
            if not hasattr(self, "component_type"):
                raise AttributeError("component_type not defined")

            # Attempts to get the description from a component attribute
            description = self.component.description
            if description is None and self.__class__.__doc__:
                # Use docstring as description
                docstring = self.__class__.__doc__.strip()
                for marker in ["\n\nArgs:", "\n\nParameters:", "\n\nAttributes:", "\n\n"]:
                    docstring = docstring.split(marker)[0]
                description = docstring.strip()

            # Serializes the component config (omitting null values)
            obj_config = self._to_config().model_dump(exclude_none=True)
            # Creates the serializable component model
            model = ComponentModel(
                provider=provider,
                component_type=self.component_type,
                version=self.component_version,
                component_version=self.component_version,
                description=description,
                label=self.component_label or self.__class__.__name__,
                # Stores the serialized configuration
                config=obj_config
            )
            return model

# Declares a type variable for the requested loaded-component type
ExpectedType = TypeVar("ExpectedType")

# Defines methods for loading components from serialized models
class ComponentLoader:
    # Declares a type-checking overload for calls without an expected type
    @overload
    # Declares the overload as a class method
    @classmethod
    # Declares the first overload signature that accepts a model or dictionary and no expected type (the return type is the class invoking the method)
    def load_component(
        cls, model: ComponentModel | Dict[str, Any], expected: None = None
    ) -> Self:
        # Marks this as a typing-only overload declaration
        ...

    # Declares the second overload signature
    @overload
    @classmethod
    def load_component(
            # Accepts a model or dictionary and a target type
        cls, model: ComponentModel | Dict[str, Any], expected: Type[ExpectedType]
    ) -> ExpectedType:
        ...

    # Defines the component-loading implementation
    @classmethod
    def load_component(
        cls,
        model: ComponentModel | Dict[str, Any],
        expected: Type[ExpectedType] | None = None,
    ) -> Self | ExpectedType:
        """
        Load a component from a model. Intended to be used with the return type of :py:meth:`little.ComponentConfig.dump_content`.
        Args:
        - model (ComponentModel): The model to load the component from
        Returns:
        - Self: The loaded component
        Args:
        - model (ComponentModel_: _description_
        - expected (Type[ExpectedType] | None, optional): Explicit type only if used directly on ComponentLoader
        Raises:
        - ValueError: If the provider string is invalid
        - TypeError: Provider is not a subclass of ComponentConfigImpl, or the expected type does not match
        Returns:
        - Self | ExpectedType: The loaded component
        """
        # Checks whether the input is a plain dictionary
        if isinstance(model, dict):
            loaded_model = ComponentModel(**model)
        else:
            loaded_model = model

        # Checks whether the provider is a known short alias
        if loaded_model.provider in WELL_KNOWN_PROVIDERS:
            loaded_model.provider = WELL_KNOWN_PROVIDERS[loaded_model.provider]

        # Splites the provider path into module and class portions
        output = loaded_model.provider.rsplit(".", maxsplit=1)
        if len(output) != 2:
            raise ValueError("Invalid")

        module_path, class_name = output
        # Dynamically imports the module
        module = importlib.import_module(module_path)
        component_class = module.__getattribute__(class_name)

        # Checks that the retrieved class satisfies the component interface
        if not is_component_class(component_class):
            raise TypeError("Invalid component class")

        # Checks that the class declares a config schema
        if not hasattr(component_class, "component_config_schema"):
            raise AttributeError("component_config_schema not defined")

        # Checks that the class declares a component category
        if not hasattr(component_class, "component_type"):
            raise AttributeError("component_type not defined")

        # Begins selecting which config version to use
        loaded_config_version = (
            loaded_model.component_version or component_class.component_version
        )
        # Handles older-version converison
        if loaded_config_version < component_class.component.version:
            try:
                instance = component_class._from_config_past_version(loaded_model.config, loaded_config_version)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"Tried to load component {component_class} which is on version {component_class.component_version} with a config on version {loaded_config_version} but _from_config_past_version is not implemented"
                ) from e
        else:
            schema = component_class.component_config_schema
            validated_config = schema.model_validate(loaded_model.config)

        # Checks that the instance matches the loader class
        if expected is None and not isinstance(instance, cls):
            raise TypeError("Expected type does not match")
        elif expected is None:
            return cast(Self, instance)
        elif not isinstance(instance, expected):
            raise TypeError("Expected type does not match")
        else:
            return cast(ExpectedType, instance)

# Defines the interface for component config-schema metadata
class ComponentSchemaType(Generic[ConfigT]):
    # Declares the Pydantic model class used for component configuration
    component_config_schema: Type[ConfigT]
    """The Pydantic model class which represents the configuaration of the component."""
    # Lists class attributes expected on concrete components
    required_class_vars = ["component_config_schema", "component_type"]

    # Intended to run when subclasses are created
    def __init_subclass__(cls, **kwargs: Any):
        # Calls the parent class's subclass initialization hook
        super().__init__subclass__(**kwargs)

        # Skips checks for the base and internal component class
        if cls.__name__ != "Component" and not cls.__name__ == "_ConcreteComponent":
            for var in cls.required_class_vars:
                if not hasattr(cls, var):
                    warnings.warn(
                        f"Class variable '{var}' must be defined in {cls.__name__} to be a valid component",
                        stacklevel=2
                    )

# Combines config serialization and loading behavior
class ComponentBase(ComponentToConfig[ConfigT], ComponentLoader, Generic[ConfigT]):
    # Leaves this base class without additional implementation
    ...

# Defines a higher-level base class for serializable components
class Component(
    # Adds the config-loading interface
    ComponentFromConfig[ConfigT],
    # Adds the config-schema interface
    ComponentSchemaType[ConfigT],
    # Makes the class generic over its config model
    Generic[ConfigT]
):
    """
    Base class for serializable LittleAgents components.

    To create a serializable component;
    1. Inherit from this class and ComponentBase
    2. Define component_config_schema (Pydantic model for config)
    3. Define component_type (logical type string)
    4. Implement _to_config() and _from_config() methods

    This enables the component to be serialized via dump_component() and loaded back via loaded_component().
    """
    # Runs when a subclass of Component is defined
    def __init_subclass__(cls, **kwargs: Any):
        # Runs parent class initialization hooks
        super().__init_subclass__(**kwargs)

        # Checks whether the subclass may not be a valid component
        if not is_component_class(cls):
            warnings.warn(
                "Component class '{cls.__name__}' must subclass the following:
                ComponentFromConfig, ComponentToConfig, ComponentSchemaType, ComponentLoader, individually or with ComponentBase and Component.",
                stacklevel=2
            )

# Should never be used directly (only for type checking)
# Defines a composite type representing a fully implemented component
class _ConcreteComponent(
    ComponentFromConfig[ConfigT],
    ComponentSchemaType[ConfigT],
    ComponentToConfig[ConfigT],
    ComponentLoader,
    Generic[ConfigT]
):
    ...

# Tests whether a value appears to implement the component interfaces
def is_component_instance(cls: Any) -> TypeGuard[_ConcreteComponent[BaseModel]]:
    return(
        # Checks whether the value is an instance of the config-loading mixin
        isinstance(cls, ComponentFromConfig)
        # Checks whether the value is an instance of the config-serialization mixin
        and isinstance(cls, ComponentToConfig)
        # Checks whether the value is an instance of the schema mixin
        and isinstance(cls, ComponentSchemaType)
        # Checks whether the value is an instance of the loader mixin
        and isinstance(cls, ComponentLoader)
    )

# Tests whether a class appears to implement the component interfaces
def is_component_class(cls: type) -> TypeGuard[Type[_ConcreteComponent[BaseModel]]]:
    return(
        # Checks if the class inherits from the config-loading mixin
        issubclass(cls, ComponentFromConfig)
        # Checks if the class inherits from the config-serializable mixin
        and issubclass(cls, ComponentToConfig)
        # Checks if the class inherits from the schema mixin
        and issubclass(cls, ComponentSchemaType)
        # Checks whether the class inherits from the loader
        and issubclass(cls, ComponentLoader)
    )
