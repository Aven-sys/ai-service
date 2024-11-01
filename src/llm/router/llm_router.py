from fastapi import APIRouter
from common.util import pydantic_util
from ..util.langchain_pydantic_model_generator import create_pydantic_model_from_config, print_model_fields
import json

# Request payload
from ..payload.request.llm_chat_request_dto import LLMChatRequestDto

router = APIRouter(
    prefix="/api/llm",
)


@router.post("")
async def llm_chat(llm_chat_request_dto: LLMChatRequestDto):
    if llm_chat_request_dto.structure_output_config:
        pydantic_config = llm_chat_request_dto.structure_output_config
        # model = pydantic_util.create_recursive_dynamic_pydantic_model(pydantic_config)
        # Print the model schema to view structure and fields
        # for field_name, field_info in model.model_fields.items():
        # print(f"  Field: {field_name}")
        # print(f"  Default: {field_info.default}")
        # print(f"  Description: {field_info.description}")
        # print()

        # print(json.dumps(pydantic_config))
#         config_json = """
# [
#     {"name": "storeName", "type": "str", "defaultValue": "BestShop", "description": "The name of the store", "nestedConfig": []},
#     {"name": "location", "type": "str", "defaultValue": "Downtown", "description": "The store's location", "nestedConfig": []},
#     {"name": "customers", "type": "List[Dict]", "defaultValue": [], "description": "List of customers", "nestedConfig": [
#         {"name": "id", "type": "int", "defaultValue": 1, "description": "Customer ID", "nestedConfig": []},
#         {"name": "name", "type": "str", "defaultValue": "John Doe", "description": "Customer name", "nestedConfig": []},
#         {"name": "address", "type": "Dict", "defaultValue": {}, "description": "Customer address", "nestedConfig": [
#             {"name": "street", "type": "str", "defaultValue": "123 Main St", "description": "Street address", "nestedConfig": []}
#         ]}
#     ]}
# ]
# """
        DynamicModel1 = create_pydantic_model_from_config(json.dumps(pydantic_config))
         # Print the model schema to view structure and fields
        print_model_fields(DynamicModel1)
        # for field_name, field_info in DynamicModel1.model_fields.items():
        #     print(f"  Field: {field_name}")
        #     print(f"  Default: {field_info.default}")
        #     print(f"  Description: {field_info.description}")
        #     print()

    return {"message": "Hello World"}
