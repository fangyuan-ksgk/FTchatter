from utils import *
import streamlit as st
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import your existing functions and any necessary modules
# from your_module import get_response_from_base, get_response_with_patch, formatting_query_prompt, tokenizer

# Streamlit app
st.title("Chat with Base and FT Models")

# Initialize session state
if 'message_history_base' not in st.session_state:
    st.session_state.message_history_base = []
if 'message_history_ft' not in st.session_state:
    st.session_state.message_history_ft = []

# Chat interface
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Add user message to both histories
    st.session_state.message_history_base.append({"role": "user", "content": user_input})
    st.session_state.message_history_ft.append({"role": "user", "content": user_input})

    # System prompt
    sys_prompt = "Roleplay as a Filipino named Maria. You are NOT an insurance agent. Do NOT offer insurance product."

    # Get responses using your existing functions
    format_prompt_base = formatting_query_prompt(st.session_state.message_history_base, sys_prompt, tokenizer)
    format_prompt_ft = formatting_query_prompt(st.session_state.message_history_ft, sys_prompt, tokenizer)

    response_base = get_response_from_base(format_prompt_base, do_print=False)
    response_ft = get_response_with_patch(format_prompt_ft, do_print=False)

    # Add responses to histories
    st.session_state.message_history_base.append({"role": "assistant", "content": response_base})
    st.session_state.message_history_ft.append({"role": "assistant", "content": response_ft})

    # Cap conversation history
    cap_rounds = 3
    st.session_state.message_history_base = st.session_state.message_history_base[-(2*cap_rounds):]
    st.session_state.message_history_ft = st.session_state.message_history_ft[-(2*cap_rounds):]

# Display chat history
col1, col2 = st.columns(2)

with col1:
    st.subheader("Base Model")
    for message in st.session_state.message_history_base:
        if message['role'] == 'user':
            st.text_input("You:", value=message['content'], key=f"base_user_{random.randint(0, 10000)}", disabled=True)
        else:
            st.text_area("Base Agent:", value=message['content'], key=f"base_agent_{random.randint(0, 10000)}", disabled=True)

with col2:
    st.subheader("FT Model")
    for message in st.session_state.message_history_ft:
        if message['role'] == 'user':
            st.text_input("You:", value=message['content'], key=f"ft_user_{random.randint(0, 10000)}", disabled=True)
        else:
            st.text_area("FT Agent:", value=message['content'], key=f"ft_agent_{random.randint(0, 10000)}", disabled=True)