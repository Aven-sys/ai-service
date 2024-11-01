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