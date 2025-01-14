from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import streamlit as st

# Set page configuration first, before any other Streamlit commands
st.set_page_config(
    page_title="טופס 101",
)

from supabase import Client
from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from ai_agent import ai_agent, AIDeps
from dotenv import load_dotenv

load_dotenv()

# Get message history limit from environment variable, default to 2 if not set
MESSAGE_HISTORY_LIMIT = int(os.getenv("MESSAGE_HISTORY_LIMIT", 3))

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Add custom CSS for RTL support (CSS remains the same)
st.markdown("""
    <style>
        .stApp {
            direction: rtl;
        }
        .main .block-container {
            direction: rtl;
            text-align: right;
        }
        .stChatMessage {
            direction: rtl;
            text-align: right;
        }
        .stChatInputContainer {
            direction: rtl;
        }
        p, h1, h2, h3 {
            direction: rtl;
            text-align: right;
        }
        div[data-testid="stMarkdownContainer"] {
            direction: rtl;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)


def maintain_message_history(messages: list) -> list:
    """
    Maintain only the last N messages in the chat history, where N is defined by MESSAGE_HISTORY_LIMIT.
    """
    if len(messages) > MESSAGE_HISTORY_LIMIT:
        return messages[-MESSAGE_HISTORY_LIMIT:]
    return messages


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    """
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str):
    """Run the agent with streaming text support for RTL."""
    deps = AIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    async with ai_agent.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages,  # Using current messages
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(f"<div dir='rtl'>{partial_text}</div>", unsafe_allow_html=True)

        filtered_messages = [msg for msg in result.new_messages()
                             if not (hasattr(msg, 'parts') and
                                     any(part.part_kind == 'user-prompt' for part in msg.parts))]

        # Update messages with history limit
        new_messages = maintain_message_history(filtered_messages)
        st.session_state.messages = new_messages

        # Add the current response
        response = ModelResponse(parts=[TextPart(content=partial_text)])
        st.session_state.messages = maintain_message_history(st.session_state.messages + [response])


async def main():
    st.markdown("<h1 style='text-align: right; direction: rtl;'>טופס 101</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: right; direction: rtl;'>שאל כל שאלה בנוגע לטופס 101</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display current messages (which will only be the last two)
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("מה השאלה שלך?")

    if user_input:
        # Create new user message
        new_message = ModelRequest(parts=[UserPromptPart(content=user_input)])

        # Update messages list with the new message while maintaining only last two
        st.session_state.messages = maintain_message_history(st.session_state.messages + [new_message])

        with st.chat_message("user"):
            st.markdown(f"<div dir='rtl'>{user_input}</div>", unsafe_allow_html=True)

        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())