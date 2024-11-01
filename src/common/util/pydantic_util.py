from typing import Any, Type, Dict, List
from ..payload.pydantic_config import PydanticConfig
from pydantic import BaseModel, Field, create_model


def get_python_type(type_name: str) -> Type:
    return {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }.get(type_name, Any)


def create_pydantic_model_from_config(
    pydantic_config: List[PydanticConfig], model_name: str = "DynamicConfigModel"
) -> Type[BaseModel]:
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