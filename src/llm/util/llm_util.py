from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from datetime import date
from decimal import Decimal

class llm():
    def __init__(self, model_name, system_Prompt, max_error_allowed, is_structured_output, structure_output_config):
        self.model_name = model_name
        self.system_Prompt = system_Prompt
        self.max_error_allowed = max_error_allowed
        self.is_structured_output = is_structured_output
        self.structure_output_config = structure_output_config

    