from langchain.memory import ConversationTokenBufferMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_memory import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Optional, Union
from .langchain_pydantic_model_generator import (
    create_pydantic_model_from_config,
    print_pydantic_model,
    print_pydantic_instance,
)
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage
import json
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Dictionary to store memory instances for each session
session_histories: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str, memory_type: str = "chat") -> BaseChatMessageHistory:
    """Get or create a chat history for a session."""
    if session_id not in session_histories:
        if memory_type == "chat":
            session_histories[session_id] = ChatMessageHistory()
            # session_histories[session_id] = InMemoryHistory()
        elif memory_type == "token":
            llm = ChatOpenAI(temperature=0)
            session_histories[session_id] = ConversationTokenBufferMemory(
                llm=llm,
                max_token_limit=1000,
                return_messages=True,  # Important: Must return messages for RunnableWithMessageHistory
                memory_key="history"   # Must match the MessagesPlaceholder variable_name
            )
        elif memory_type == "summarize":
            llm = ChatOpenAI(temperature=0)
            session_histories[session_id] = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=1000,
                return_messages=True,  # Important: Must return messages for RunnableWithMessageHistory
                memory_key="history"   # Must match the MessagesPlaceholder variable_name
            )
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
    
    return session_histories[session_id]

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []