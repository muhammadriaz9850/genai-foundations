import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from ai_engine import ChatBotEngine

# --- Page Configuration ---
st.set_page_config(
    page_title="GenAI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for a more polished look
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .main {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Initialization ---
if "bot_engine" not in st.session_state:
    st.session_state.bot_engine = ChatBotEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 AI Configuration")
    st.info("This app uses a custom LangChain engine powered by Ollama.")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### Model Settings")
    # Temperature slider
    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    # Use getattr to avoid crash if the session_state object was created before the class change
    current_temp = getattr(st.session_state.bot_engine, 'temperature', 0.1)
    if temp != current_temp:
        st.session_state.bot_engine.update_temperature(temp)

    st.divider()
    st.markdown("### About")
    st.write("A professional Streamlit interface for the GenAI Foundations project.")

# --- Main UI ---
st.title("GenAI Chatbot")
st.caption("Experience a modern, streaming AI assistant powered by LangChain and Ollama.")

# Display chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to state and display
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("Thinking..."):
                # Use the engine's stream method
                for chunk in st.session_state.bot_engine.get_response_stream(st.session_state.messages):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            # Add assistant message to history
            st.session_state.messages.append(AIMessage(content=full_response))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Optional: Log error here
