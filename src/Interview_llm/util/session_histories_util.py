from langchain.memory import ChatMessageHistory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_memory import BaseChatMessageHistory
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

# Dictionary to store history for each session
session_histories = {}

def get_session_history(session_id: str, memory_type: str = "chat") -> BaseChatMessageHistory:
    # Initialize session with the chosen memory type
    if session_id not in session_histories:
        if memory_type == "chat":
            session_histories[session_id] = ChatMessageHistory()
        elif memory_type == "token":
            session_histories[session_id] = ConversationTokenBufferMemory(token_limit=1000)  # Adjust token limit as needed
        elif memory_type == "summarize":
            session_histories[session_id] = ConversationSummaryBufferMemory()
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
    return session_histories[session_id]

