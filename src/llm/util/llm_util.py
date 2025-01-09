from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from datetime import date
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
from .langchain_pydantic_model_generator import (
    create_pydantic_model_from_config,
    print_pydantic_model,
    print_pydantic_instance,
)
import json
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class LLM:
    model_cache = {}

    def __init__(
        self,
        input,
        model_name,
        system_Prompt,
        max_error_allowed,
        is_structured_output,
        structure_output_config,
        prompt_type=None,
    ):
        self.model_name = model_name
        self.system_Prompt = system_Prompt
        self.max_error_allowed = max_error_allowed
        self.is_structured_output = is_structured_output
        self.structure_output_config = structure_output_config
        self.model = self.initialize_model()
        self.prompt_type = prompt_type
        self.input = input
        self.parser = self.initialize_parser()

    def initialize_model(self):
        # Check if the model is already in the cache
        if self.model_name not in LLM.model_cache:
            # If not, instantiate a new model and store it in the cache
            model = ChatOpenAI(model=self.model_name, temperature=0.5)
            LLM.model_cache[self.model_name] = model
        # Return the model from the cache
        return LLM.model_cache[self.model_name]
    # def initialize_prompt(self):
    #     # Initialize the prompt template, with an optional system prompt
    #     if self.prompt_type == "chat":
    #         return ChatPromptTemplate([SystemMessage(content=self.system_prompt)])
    #     else:
    #         template_settings = {
    #             "template": self.system_Prompt,
    #             "input_variables": [key for key, value in self.input.items()],
    #         }
    #     # Conditionally add partial_variables if structured output is required
    #     if self.is_structured_output:
    #         template_settings["partial_variables"] = {
    #             "format_instructions": self.parser.get_format_instructions()
    #         }
    #     return PromptTemplate(**template_settings)
    
    def initialize_prompt(self):
    # Initialize the prompt template, with an optional system prompt
        if self.prompt_type == "chat":
            self.prompt_template = ChatPromptTemplate([SystemMessage(content=self.system_Prompt)])
        else:
            template_settings = {
                "template": self.system_Prompt,
                "input_variables": [key for key, value in self.input.items()],
            }
            # Conditionally add partial_variables if structured output is required
            if self.is_structured_output:
                template_settings["partial_variables"] = {
                    "format_instructions": self.parser.get_format_instructions()
                }
            self.prompt_template = PromptTemplate(**template_settings)
        return self.prompt_template

    def initialize_parser(self):
        # If structured output is required, configure the parser based on structure_output_config
        if isinstance(self.structure_output_config, list) and self.is_structured_output:
            pydantic_model_config = create_pydantic_model_from_config(
                json.dumps(self.structure_output_config)
            )
            # print_pydantic_model(pydantic_model_config)
            return PydanticOutputParser(pydantic_object=pydantic_model_config)

        return None

    def create_messages(self, input_data: Dict[str, str]) -> List[HumanMessage]:
        # Generate messages for the chat interaction
        messages = [self.prompt_template.format(input_data)]
        messages.append(HumanMessage(content=input_data.get("message", "")))
        return messages

    def create_chain(self):
        prompt = self.initialize_prompt()
        chain = prompt | self.model | self.parser
        return chain

    def run(self, input_data: Dict[str, str]) -> Any:
        chain = self.create_chain()

        # Resolve and print the system prompt for debugging
        # resolved_prompt = self.prompt_template.format_prompt(**input_data).to_string()
        # print("Resolved System Prompt:")
        # print(resolved_prompt)

        response = chain.invoke(input_data)
        # print_pydantic_instance(response)
        # print(type(response))
        return response
        # Execute the chat with retry mechanism based on max_error_allowed
        # retries = 0
        # while retries < self.max_error_allowed:
        #     try:
        #         response = chain.invoke(input_data)
        #         print(response)
        #         # if self.parser:
        #         #     return self.parser.parse(response)
        #         # return response
        #     except Exception as e:
        #         retries += 1
        #         if retries >= self.max_error_allowed:
        #             raise e  # Fail after max retries
