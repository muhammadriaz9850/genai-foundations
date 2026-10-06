import streamlit as st
from ai_engine import AIEngine
from langchain_core.messages import HumanMessage, AIMessage

# Page configuration for a modern, professional look
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for polished UI
st.markdown("""
    <style>
        .stChatMessage {
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .main {
            max-width: 800px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🤖 Modern AI Assistant")
    st.caption("Powered by Ollama and LangChain")

    # Initialize AI Engine
    if "ai_engine" not in st.session_state:
        st.session_state.ai_engine = AIEngine()

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = st.session_state.ai_engine.get_initial_messages()

    # Sidebar for settings and controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = st.session_state.ai_engine.get_initial_messages()
            st.rerun()

        st.divider()
        st.info("This app uses the `qwen:0.5b` model via Ollama for fast, local inference.")

    # Display chat messages (skip the system message)
    for msg in st.session_state.messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add to session state
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Generate AI response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                with st.spinner("Thinking..."):
                    # Start streaming
                    for chunk in st.session_state.ai_engine.stream_response(st.session_state.messages):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                    # Final update without cursor
                    message_placeholder.markdown(full_response)

                # Add AI response to session state
                st.session_state.messages.append(AIMessage(content=full_response))

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please ensure Ollama is running locally.")

if __name__ == "__main__":
    main()
