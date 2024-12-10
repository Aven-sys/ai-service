from typing import Any, Dict, List, Type, get_origin, get_args, Union
from pydantic import BaseModel, Field
import json
from pydantic.main import create_model
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.syntax import Syntax

class PydanticConfigModelGenerator:
    def __init__(
        self,
        name: str,
        type: str = None,
        defaultValue: Any = None,
        description: str = None,
        nestedConfig: List[dict] = None,
    ):
        self.name = name
        self.type = type
        self.default = defaultValue
        self.description = description
        self.nested_config = (
            [PydanticConfigModelGenerator(**nc) for nc in nestedConfig]
            if nestedConfig
            else []
        )

    def get_field_type(self):
        """Get the actual Python type for the field"""
        type_map = {
            "str": str,
            "int": int,
            "List[str]": List[str],
            "List[int]": List[int],
            "List[Dict]": List[Dict],
            "Dict": Dict,
        }

        if self.nested_config:
            # For nested configurations, create a nested model
            nested_model = build_model(
                self.nested_config, f"{self.name.capitalize()}Model"
            )
            return List[nested_model] if self.type == "List[Dict]" else nested_model

        return type_map.get(self.type, Any)


def build_model(
    config: List[PydanticConfigModelGenerator], model_name: str
) -> Type[BaseModel]:
    """Build a Pydantic model from configuration"""
    fields = {}
    for field in config:
        field_type = field.get_field_type()
        fields[field.name] = (
            field_type,
            Field(default=field.default, description=field.description),
        )

    return create_model(model_name, **fields)


def create_pydantic_model_from_config(
    config_json: str, model_name: str = "DynamicConfigModel"
) -> Type[BaseModel]:
    """Create a Pydantic model from JSON configuration"""
    if isinstance(config_json, str):
        config_list = json.loads(config_json)
    else:
        config_list = config_json

    config_objects = [PydanticConfigModelGenerator(**item) for item in config_list]
    return build_model(config_objects, model_name)


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
            print_model_fields(
                field_type, indent + 1
            )  # Recursive call for nested models
        print()  # Blank line for readability

def print_pydantic_model(model: Type[BaseModel], 
                          title: str = "Model Structure", 
                          include_methods: bool = False, 
                          console: Console = None,
                          _depth: int = 0) -> None:
    """
    Comprehensively print the structure of a Pydantic model with detailed nested model support.
    
    Args:
        model (Type[BaseModel]): The Pydantic model to print
        title (str, optional): Title for the model structure. Defaults to "Model Structure".
        include_methods (bool, optional): Whether to include model methods. Defaults to False.
        console (Console, optional): Rich Console instance. Defaults to None.
        _depth (int, optional): Internal depth tracking for recursion. Defaults to 0.
    """
    # Prevent excessive recursion
    if _depth > 5:
        return
    
    # Use default console if not provided
    if console is None:
        console = Console()
    
    # Create the main tree
    model_tree = Tree(f"[bold blue]{title}[/bold blue]")
    
    # Iterate through model fields
    for field_name, field_info in model.model_fields.items():
        # Determine field type
        field_type = _get_type_str(field_info.annotation)
        description = field_info.description or "No description"
        
        # Create a branch for each field
        field_branch = model_tree.add(
            f"[green]{field_name}[/green]: [yellow]{field_type}[/yellow] - [dim]{description}[/dim]"
        )
        
        # Handle nested models and list of models
        _handle_nested_types(field_info.annotation, field_branch, _depth)
    
    # Optionally include methods
    if include_methods:
        methods_branch = model_tree.add("[bold]Methods:[/bold]")
        for method_name in dir(model):
            if not method_name.startswith('_') and callable(getattr(model, method_name)):
                methods_branch.add(f"[cyan]{method_name}[/cyan]")
    
    # Print the tree in a panel
    console.print(Panel(model_tree, border_style="blue", expand=False))

def _get_type_str(annotation: Any) -> str:
    """
    Get a readable string representation of the type.
    
    Args:
        annotation: Type annotation to convert to string
    
    Returns:
        str: Readable type representation
    """
    # Handle Optional types
    if get_origin(annotation) is Union:
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            return f"Optional[{_get_type_str(args[0])}]"
    
    # Handle List types
    if get_origin(annotation) is list:
        args = get_args(annotation)
        return f"List[{_get_type_str(args[0]) if args else 'Any'}]"
    
    # Handle basic types and model types
    try:
        return annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)
    except Exception:
        return str(annotation)

def _handle_nested_types(annotation: Any, parent_branch: Tree, depth: int):
    """
    Recursively handle nested model types.
    
    Args:
        annotation: Type annotation to inspect
        parent_branch (Tree): Parent tree branch to attach nested details
        depth (int): Current recursion depth
    """
    # Handle Optional types
    if get_origin(annotation) is Union:
        args = get_args(annotation)
        annotation = [arg for arg in args if arg is not type(None)][0]
    
    # Handle List types
    if get_origin(annotation) is list:
        args = get_args(annotation)
        if args:
            annotation = args[0]
    
    # Check if it's a Pydantic model
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        # Add nested model details
        nested_branch = parent_branch.add("[bold cyan]Nested Model:[/bold cyan]")
        
        for nested_field_name, nested_field_info in annotation.model_fields.items():
            # Get nested field type
            nested_field_type = _get_type_str(nested_field_info.annotation)
            nested_description = nested_field_info.description or "No description"
            
            # Add nested field details
            nested_field_branch = nested_branch.add(
                f"[green]{nested_field_name}[/green]: [yellow]{nested_field_type}[/yellow] - [dim]{nested_description}[/dim]"
            )
            
            # Recursively handle further nested types
            if depth < 5:
                _handle_nested_types(nested_field_info.annotation, nested_field_branch, depth + 1)

def print_pydantic_instance(instance: BaseModel, 
                             title: str = "Model Instance", 
                             show_json: bool = False, 
                             console: Console = None) -> None:
    """
    Comprehensively print the data of a Pydantic model instance.
    
    Args:
        instance (BaseModel): The Pydantic model instance to print
        title (str, optional): Title for the instance. Defaults to "Model Instance".
        show_json (bool, optional): Whether to show JSON representation. Defaults to False.
        console (Console, optional): Rich Console instance. Defaults to None.
    """
    # Use default console if not provided
    if console is None:
        console = Console()
    
    # Create the main tree
    instance_tree = Tree(f"[bold blue]{title}[/bold blue]")
    
    # Recursive function to add nested data to the tree
    def add_data_to_tree(data: Any, parent_branch: Tree, field_name: str = None):
        """
        Recursively add data to the tree with formatting.
        
        Args:
            data (Any): Data to be added
            parent_branch (Tree): Parent tree branch to attach data
            field_name (str, optional): Name of the field. Defaults to None.
        """
        # Handle None values
        if data is None:
            branch = parent_branch.add(f"[dim]{field_name or 'Value'}: None[/dim]")
            return
        
        # Handle Pydantic models
        if isinstance(data, BaseModel):
            # Create a branch for the nested model
            model_branch = parent_branch.add(f"[green]{field_name or type(data).__name__}[/green]:")
            
            # Iterate through model fields
            for sub_field_name, sub_field_value in data.model_dump().items():
                add_data_to_tree(sub_field_value, model_branch, sub_field_name)
            return
        
        # Handle lists
        if isinstance(data, list):
            # Create a branch for the list
            list_branch = parent_branch.add(f"[green]{field_name or 'List'}[/green]: {len(data)} items")
            
            # Add each list item
            for i, item in enumerate(data):
                item_branch = list_branch.add(f"[yellow]Item {i}[/yellow]")
                add_data_to_tree(item, item_branch)
            return
        
        # Handle dictionaries
        if isinstance(data, dict):
            # Create a branch for the dictionary
            dict_branch = parent_branch.add(f"[green]{field_name or 'Dictionary'}[/green]: {len(data)} items")
            
            # Add each dictionary item
            for key, value in data.items():
                key_branch = dict_branch.add(f"[yellow]{key}[/yellow]")
                add_data_to_tree(value, key_branch)
            return
        
        # Handle simple types
        value_str = str(data)
        value_type = type(data).__name__
        parent_branch.add(f"[green]{field_name or 'Value'}[/green]: [yellow]{value_str}[/yellow] [{value_type}]")
    
    # Add model data to the tree
    for field_name, field_value in instance.model_dump().items():
        add_data_to_tree(field_value, instance_tree, field_name)
    
    # Print the tree in a panel
    console.print(Panel(instance_tree, border_style="blue", expand=False))
    
    # Optionally show JSON representation
    if show_json:
        console.print("\n[bold]JSON Representation:[/bold]")
        json_syntax = Syntax(
            instance.model_dump_json(indent=2), 
            "json", 
            theme="monokai", 
            line_numbers=True
        )
        console.print(json_syntax)