# Created by Claude (author: Nicholas Beaudoin)

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
import os
from typing import List
import json

# Initialize Streamlit page configuration
st.set_page_config(page_title="Chinese Language Tutor", page_icon="🇨🇳", layout="wide")

# System prompt for the tutor
SYSTEM_PROMPT = """You are a helpful Chinese language tutor specifically teaching at the HSK 4 level. 
You should:
1. Respond to student messages in Chinese (using HSK 4 level vocabulary and grammar)
2. Provide pinyin for all Chinese characters
3. Provide English translations
4. If the student writes in Chinese, correct any mistakes they make
5. Use appropriate HSK 4 vocabulary and grammar patterns in your responses
6. Be encouraging and supportive

Format your responses as JSON with the following structure:
{
    "chinese": "Chinese text",
    "pinyin": "Pinyin with tones",
    "english": "English translation",
    "corrections": "Any corrections (if applicable)",
    "explanation": "Brief explanation of grammar or vocabulary used"
}"""

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
    if 'token_count' not in st.session_state:
        st.session_state.token_count = 0
    if 'conversation_cost' not in st.session_state:
        st.session_state.conversation_cost = 0

def setup_openai():
    """Setup OpenAI API key and model."""
    # Use environment variable if set, otherwise use sidebar input
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = st.sidebar.text_input('OpenAI API Key', type='password')
        if not api_key:
            st.error("Please provide an OpenAI API key!")
            st.stop()
    
    return ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4",
        openai_api_key=api_key
    )

def parse_response(response_text: str) -> dict:
    """Parse the AI response from JSON to dict."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback if response isn't proper JSON
        return {
            "chinese": response_text,
            "pinyin": "Error parsing response",
            "english": "Error parsing response",
            "corrections": "",
            "explanation": "Error parsing the tutor's response"
        }

def format_message(message_dict: dict) -> str:
    """Format the message dictionary for display."""
    formatted = f"🈺 {message_dict['chinese']}\n\n"
    formatted += f"🔈 {message_dict['pinyin']}\n\n"
    formatted += f"🌏 {message_dict['english']}\n\n"
    
    if message_dict['corrections']:
        formatted += f"✍️ Corrections: {message_dict['corrections']}\n\n"
    
    if message_dict['explanation']:
        formatted += f"📝 Note: {message_dict['explanation']}"
    
    return formatted

def main():
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar
    st.sidebar.title("Chinese Tutor Settings")
    llm = setup_openai()
    
    # Main title
    st.title("HSK 4 Chinese Language Tutor 🎓")
    st.markdown("""
    Welcome to your Chinese language tutor! Feel free to:
    - Ask questions in English or Chinese
    - Practice writing Chinese sentences
    - Request explanations of grammar points
    - Get vocabulary help
    """)
    
    # Display conversation history
    for message in st.session_state.messages[1:]:  # Skip the system message
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                response_dict = parse_response(message.content)
                st.markdown(format_message(response_dict))
    
    # User input
    if prompt := st.chat_input("Type your message here (English or Chinese)"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                with get_openai_callback() as cb:
                    response = llm(st.session_state.messages)
                    st.session_state.token_count += cb.total_tokens
                    st.session_state.conversation_cost += cb.total_cost
                
                response_dict = parse_response(response.content)
                st.markdown(format_message(response_dict))
                st.session_state.messages.append(AIMessage(content=response.content))
    
    # Display usage metrics in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Conversation Stats")
    st.sidebar.markdown(f"Tokens Used: {st.session_state.token_count}")
    st.sidebar.markdown(f"Estimated Cost: ${st.session_state.conversation_cost:.4f}")
    
    # Reset conversation button
    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        st.session_state.token_count = 0
        st.session_state.conversation_cost = 0
        st.rerun()

if __name__ == "__main__":
    main()