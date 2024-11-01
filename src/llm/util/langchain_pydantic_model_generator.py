import json
from typing import List, Dict, Type, Any, Optional
from pydantic import BaseModel, create_model, Field

class PydanticConfigModelGenerator:
    def __init__(self, name: str, type: str = None, defaultValue: Any = None, description: str = None, nestedConfig: List[dict] = None):
        self.name = name
        self.type = self.map_type(type)  # Renaming `type` to `type_` internally
        self.default = defaultValue
        self.description = description
        self.nested_config = [PydanticConfigModelGenerator(**nc) for nc in nestedConfig] if nestedConfig else []

    @staticmethod
    def map_type(type_str: str):
        # Map string type names to actual Python types
        type_map = {
            "str": str,
            "int": int,
            "List[Dict]": List[Dict],
            "Dict": Dict,
        }
        return type_map.get(type_str, Any)

def create_pydantic_model_from_config(config_json: str, model_name: str = "DynamicConfigModel") -> Type[BaseModel]:
    """
    Parses JSON config and creates a Pydantic model.
    """
    config_list = json.loads(config_json)
    config_objects = [PydanticConfigModelGenerator(**item) for item in config_list]
    return build_model(config_objects, model_name)

def build_model(config: List[PydanticConfigModelGenerator], model_name: str) -> Type[BaseModel]:
    fields = {}
    for field in config:
        if field.nested_config:
            sub_model = build_model(field.nested_config, model_name=field.name.capitalize() + "Model")
            fields[field.name] = (sub_model, Field(default=field.default, description=field.description))
        else:
            fields[field.name] = (field.type, Field(default=field.default, description=field.description))
    return create_model(model_name, **fields)


def print_model_fields(model, indent=0):
    """
    Recursively prints out the fields, types, defaults, and descriptions of a Pydantic model, including nested fields.
    """
    indent_str = "  " * indent
    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        print(f"{indent_str}Field: {field_name}")
        print(f"{indent_str}  Type: {field_type}")
        print(f"{indent_str}  Default: {field_info.default}")
        print(f"{indent_str}  Description: {field_info.description}")
        
        # Check if the field type is a Pydantic model (for nested models)
        if hasattr(field_type, "model_fields"):
            print(f"{indent_str}  Nested Fields for {field_name}:")
            print_model_fields(field_type, indent + 1)  # Recursive call for nested models
        print()  # Blank line for readability