from fastapi import APIRouter, HTTPException
from common.util import pydantic_util
from common.util.langchain_pydantic_model_generator import (
    print_pydantic_instance,
    print_model_fields,
)
from ..util.langchain_pydantic_model_generator import (
    create_pydantic_model_from_config,
    print_model_fields,
)
from ..service import interview_llm_service

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from uuid import uuid4
from ..util.session_histories_util import get_session_history, session_histories
from langchain_core.runnables.history import RunnableWithMessageHistory
import json

# Request payload
from ..payload.request.llm_chat_request_dto import LLMChatRequestDto

router = APIRouter(
    prefix="/api/llm-interview",
)


class InterviewInput(BaseModel):
    session_id: str
    user_input: str


# Define Pydantic model for structured output
class InterviewOutput(BaseModel):
    # full_output: str = Field(..., description="The full conversation output with the most current llm response to it")
    interviewer_output: str = Field(
        ...,
        description="The interviewer output with the most current LLM response to it. The question should be in the output when the interviewer is asking a question.",
    )
    question: Optional[str] = Field(
        None,
        description="The current question being asked by the interviewer. This is just to store the question for reference.",
    )
    answer: Optional[str] = Field(
        None,
        description="The interviewee's answer to the current question. Do not paraphrase or summarize the answer. This is just to store the answer for reference.",
    )
    is_done: bool = Field(
        ..., description="Indicates if the interview process is complete"
    )
    summary: Optional[str] = Field(
        None,
        description="A summary of all answers provided by the interviewee at the end of the interview",
    )
    chat_history: Optional[str] = Field(
        None,
        description="The full chat history of the interview. Interview output and user input.",
    )


# System prompt that includes all questions and instructions
system_prompt = """
You are an interviewer conducting a structured interview. Ask one question at a time, then wait for the interviewee's response before proceeding to the next question. 
Greet the interviewee first politely. Then start the interview by asking the following questions:
1. How old are you?
2. What is your favorite movie?
3. What is your favorite food?
4. What is your highest education level?
5. What is your favorite animal?

After each answer, provide brief feedback if appropriate, then stop. Wait for the interviewee’s next input before asking the next question.

{format_instructions}
"""

# Initialize the parser
parser = PydanticOutputParser(pydantic_object=InterviewOutput)

# Initalize the chat prompt template
# prompt_template = PromptTemplate(
#     input_variables=["history"],
#     template=system_prompt + "\n{history}",
#     partial_variables={
#         "format_instructions": parser.get_format_instructions(),
#     },
# )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Combine the prompt, LLM, and parser into a chain
# chain = prompt | llm | parser
chain = prompt | llm

# Wrap the chain with RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


@router.post("/start-interview")
async def start_interview(memory_type: str = "chat"):
    session_id = str(uuid4())

    # Initialize the session with the specified memory type
    get_session_history(session_id, memory_type=memory_type)

    # Start the interview with an empty input (LLM will handle the flow)
    result = chain_with_history.invoke(
        {"input": "Let's start the interview"},
        config={"configurable": {"session_id": session_id}},
    )
    interview_output = InterviewOutput(**json.loads(result.content))

    return {"session_id": session_id, "response": interview_output}


@router.post("/interview")
async def interview(interview_input: InterviewInput):
    # Check if session exists
    if interview_input.session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    # Retrieve and print the chat history to verify it's being retained
    history = session_histories[interview_input.session_id]
    print("Current chat history:", history)

    # Invoke the chain with user response and session history
    result = chain_with_history.invoke(
        {"input": interview_input.user_input},
        config={"configurable": {"session_id": interview_input.session_id}},
    )

    interview_output = InterviewOutput(**json.loads(result.content))
    print("Current chat history after user input:", result)

    # # Determine if interview is done based on `is_done` in result
    # if result.is_done:
    #     summary = (
    #         "Summary of the interview:\n" + result.summary
    #         if result.summary
    #         else "No summary available."
    #     )
    #     return {"response": result.full_output, "is_done": True, "summary": summary}

    return {"response": interview_output, "is_done": False}


# End the interview and retrieve the full history
@router.post("/end-interview")
async def end_interview(session_id: str):
    if session_id not in session_histories:
        raise HTTPException(status_code=404, detail="Session not found")

    # Retrieve full conversation history
    history = session_histories[session_id].get_all_messages()
    full_history = [{"role": msg.role, "content": msg.content} for msg in history]

    # Clean up memory after retrieving history
    del session_histories[session_id]

    return {"transcript": full_history}
