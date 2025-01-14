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

# Import all the message part classes
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

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), )
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Add custom CSS for RTL support
st.markdown("""
    <style>
        /* Global RTL settings */
        .stApp {
            direction: rtl;
        }

        /* Fix header alignment */
        .main .block-container {
            direction: rtl;
            text-align: right;
        }

        /* Adjust chat messages for RTL */
        .stChatMessage {
            direction: rtl;
            text-align: right;
        }

        /* Fix input field alignment */
        .stChatInputContainer {
            direction: rtl;
        }

        /* Ensure Hebrew text is properly aligned */
        p, h1, h2, h3 {
            direction: rtl;
            text-align: right;
        }

        /* Fix streamlit components alignment */
        div[data-testid="stMarkdownContainer"] {
            direction: rtl;
            text-align: right;
        }
    </style>
""", unsafe_allow_html=True)


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
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
            message_history=st.session_state.messages[:-1],
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(f"<div dir='rtl'>{partial_text}</div>", unsafe_allow_html=True)

        filtered_messages = [msg for msg in result.new_messages()
                             if not (hasattr(msg, 'parts') and
                                     any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def main():
    st.markdown("<h1 style='text-align: right; direction: rtl;'>טופס 101</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: right; direction: rtl;'>שאל כל שאלה בנוגע לטופס 101</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("מה השאלה שלך?")

    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        with st.chat_message("user"):
            st.markdown(f"<div dir='rtl'>{user_input}</div>", unsafe_allow_html=True)

        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)


if __name__ == "__main__":
    asyncio.run(main())