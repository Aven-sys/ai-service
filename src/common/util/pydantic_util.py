from typing import Any, Type, Dict, List, Optional, Union
from ..payload.pydantic_config import PydanticConfig
from pydantic import BaseModel, Field, create_model, RootModel
import json
from datetime import datetime


def get_python_type(type_name: str) -> Type:
    return {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict
    }.get(type_name, Any)


def create_pydantic_model_from_config(pydantic_config: List[PydanticConfig], model_name: str = "DynamicConfigModel") -> Type[BaseModel]:
    """
    Create a Pydantic model dynamically from a list of PydanticConfig objects.

    Args:
        pydantic_config (List[PydanticConfig]): List of configuration items.
        model_name (str): Name of the dynamically created Pydantic model. Default is 'DynamicConfigModel'.

    Returns:
        Type[BaseModel]: A dynamically created Pydantic model class.
    """

    # Construct the fields for `create_model`
    fields: Dict[str, tuple] = {
        field.name: (
            get_python_type(field.type),
            Field(default=field.default, description=field.description),
        )
        for field in pydantic_config
    }

    # Dynamically create the Pydantic model
    DynamicModel = create_model(model_name, **fields)

    return DynamicModel


def create_recursive_model(field_name: str, config: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Type[BaseModel]:
    """
    Recursively create a Pydantic model for nested fields, including lists of objects.
    
    Args:
        field_name (str): Name for the nested model.
        config (Union[Dict[str, Any], List[Dict[str, Any]]]): Configuration dictionary or list for the nested model.
        
    Returns:
        Type[BaseModel]: A dynamically created nested Pydantic model.
    """
    fields = {}

    # Handle the case where config is a list of objects
    if isinstance(config, list):
        if config and isinstance(config[0], dict):
            item_model = create_recursive_model(f"{field_name}_item", config[0])
            return RootModel[List[item_model]]
        else:
            item_type = get_python_type(config[0]) if config else Any
            return RootModel[List[item_type]]

    # Handle the case where config is a dictionary of fields
    for key, value in config.items():
        if isinstance(value, dict) and "type" in value:
            field_type = value["type"]

            if field_type == "dict":
                nested_model = create_recursive_model(key, value.get("default", {}))
                fields[key] = (Optional[nested_model], Field(description=value.get("description", "")))

            elif field_type == "list" and "items" in value:
                item_model = create_recursive_model(f"{key}_item", value["items"])
                fields[key] = (Optional[List[item_model]], Field(description=value.get("description", "")))

            else:
                python_type = get_python_type(field_type)
                fields[key] = (Optional[python_type], Field(default=value.get("default"), description=value.get("description", "")))
        else:
            python_type = get_python_type(value) if isinstance(value, str) else get_python_type(value.get("type", "str"))
            fields[key] = (Optional[python_type], Field(default=value.get("default") if isinstance(value, dict) else None, description=value.get("description", "")) if isinstance(value, dict) else None)

    return create_model(field_name, **fields)

def create_recursive_dynamic_pydantic_model(config: Union[Dict[str, Any], List[Dict[str, Any]]], model_name: str = "DynamicConfigModel") -> Type[BaseModel]:
    """
    Create a Pydantic model dynamically, handling multiple levels of nested objects and lists.
    
    Args:
        config (Union[Dict[str, Any], List[Dict[str, Any]]]): Configuration for the root model.
        model_name (str): Name of the root model.
        
    Returns:
        Type[BaseModel]: A dynamically created Pydantic model with nested structures.
    """
    if isinstance(config, list):
        if config and isinstance(config[0], dict):
            item_model = create_recursive_model(f"{model_name}_item", config[0])
            return RootModel[List[item_model]]
        else:
            item_type = get_python_type(config[0]) if config else Any
            return RootModel[List[item_type]]
    return create_recursive_model(model_name, config)


class ModelGenerator:
    def __init__(self):
        self.model_cache = {}
        self.counter = 0

    def _generate_model_name(self, prefix: str = "DynamicModel") -> str:
        """Generate a unique model name"""
        self.counter += 1
        return f"{prefix}{self.counter}"

    def _infer_type(self, value: Any) -> type:
        """Infer Python type from JSON value"""
        if value is None:
            return Any
        if isinstance(value, bool):
            return bool
        if isinstance(value, int):
            return int
        if isinstance(value, float):
            return float
        if isinstance(value, str):
            # Try parsing as datetime
            try:
                datetime.fromisoformat(value)
                return datetime
            except ValueError:
                return str
        if isinstance(value, list):
            if value:
                # Infer type from first element
                element_type = self._process_value(value[0])
                return List[element_type]
            return List[Any]
        if isinstance(value, dict):
            return self._create_model_from_dict(value)
        return Any

    def _process_value(self, value: Any) -> type:
        """Process a value and return its corresponding type"""
        if isinstance(value, dict):
            return self._create_model_from_dict(value)
        if isinstance(value, list):
            if value:
                element_type = self._process_value(value[0])
                return List[element_type]
            return List[Any]
        return self._infer_type(value)

    def _create_model_from_dict(self, data: Dict[str, Any]) -> type:
        """Create a Pydantic model from a dictionary"""
        # Generate a hash for the dictionary to use as cache key
        dict_hash = str(hash(json.dumps(data, sort_keys=True)))
        
        # Return cached model if exists
        if dict_hash in self.model_cache:
            return self.model_cache[dict_hash]

        field_definitions = {}
        for field_name, field_value in data.items():
            field_type = self._process_value(field_value)
            field_definitions[field_name] = (field_type, ...)

        model = create_model(
            self._generate_model_name(),
            **field_definitions
        )

        # Cache the model
        self.model_cache[dict_hash] = model
        return model

    def create_model(self, config: Union[Dict[str, Any], List[Dict[str, Any]]]) -> type:
        """
        Create a Pydantic model from a JSON config.
        
        Args:
            config: Either a dictionary or a list of dictionaries
            
        Returns:
            A Pydantic model class or List[model] if input is a list
        """
        if isinstance(config, list):
            if not config:
                return List[Any]
            # Create model from first item and wrap in List
            element_type = self._create_model_from_dict(config[0])
            return List[element_type]
        return self._create_model_from_dict(config)