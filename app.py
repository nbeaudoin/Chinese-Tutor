# Created by Claude Sonnet 3.5 (author: Nicholas Beaudoin)
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import OpenAICallbackHandler
import os
from typing import List, Dict, Any, Optional
import json
from dotenv import load_dotenv
from datetime import datetime
import tiktoken
from contextlib import contextmanager
import traceback

# Load environment variables from .env file
load_dotenv()

# Initialize Streamlit page configuration
st.set_page_config(page_title="Chinese Language Tutor", page_icon="🇨🇳", layout="wide")

# HSK 4 vocabulary and grammar examples for the tutor's reference
HSK4_REFERENCE = {
    "vocab_examples": [
        "建议", "根据", "要求", "一般来说", "比如", "关系", "参加",
        "经验", "实际", "态度", "表示", "发生", "方便", "符合"
    ],
    "grammar_patterns": [
        "是...的", "越...越...", "虽然...但是...", "不管...都...",
        "除了...以外...", "一边...一边...", "从来不", "要是...就..."
    ]
}

# Enhanced system prompt for the tutor
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
    "chinese": "Chinese text using HSK 4 vocabulary",
    "pinyin": "Pinyin with tones",
    "english": "English translation",
    "corrections": "Corrections for student mistakes (if any)",
    "explanation": "Grammar and vocabulary explanations",
    "tips": "Learning suggestions or mnemonics (optional)"
}"""

class TokenTracker:
    """Custom token tracking class with detailed statistics."""
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0
        self.session_start = datetime.now()
        self.history: List[Dict[str, Any]] = []
        self._encoder = tiktoken.encoding_for_model("gpt-4")

    def add_interaction(self, prompt_tokens: int, completion_tokens: int, cost: float):
        """Record a new interaction's token usage."""
        interaction = {
            'timestamp': datetime.now(),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'cost': cost
        }
        self.history.append(interaction)
        self.update_totals(interaction)

    def update_totals(self, interaction: Dict[str, Any]):
        """Update running totals with new interaction data."""
        self.prompt_tokens += interaction['prompt_tokens']
        self.completion_tokens += interaction['completion_tokens']
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.total_cost += interaction['cost']

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a piece of text."""
        try:
            return len(self._encoder.encode(text))
        except Exception as e:
            st.warning(f"Token estimation failed: {str(e)}")
            return 0

    def get_session_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for the current session."""
        try:
            return {
                'session_duration': str(datetime.now() - self.session_start).split('.')[0],
                'total_interactions': len(self.history),
                'total_tokens': self.total_tokens,
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'average_tokens_per_interaction': self.total_tokens / len(self.history) if self.history else 0,
                'total_cost': self.total_cost,
                'average_cost_per_interaction': self.total_cost / len(self.history) if self.history else 0
            }
        except Exception as e:
            st.error(f"Error calculating session stats: {str(e)}")
            return {}

def validate_system_prompt() -> bool:
    """Validate the system prompt structure and content."""
    try:
        if "{" not in SYSTEM_PROMPT or "}" not in SYSTEM_PROMPT:
            st.error("Missing JSON format example in system prompt")
            return False
        return True
    except Exception as e:
        st.error(f"Error validating system prompt: {str(e)}")
        return False

def initialize_session_state():
    """Initialize session state variables with enhanced tracking."""
    if not validate_system_prompt():
        st.error("System prompt validation failed. Please check the prompt configuration.")
        st.stop()

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
    if 'token_tracker' not in st.session_state:
        st.session_state.token_tracker = TokenTracker()
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    if 'interaction_history' not in st.session_state:
        st.session_state.interaction_history = []

def parse_response(response_text: str) -> Dict[str, str]:
    """Parse the AI response from JSON to dict with error handling."""
    try:
        response_dict = json.loads(response_text)
        required_fields = ["chinese", "pinyin", "english", "corrections", "explanation"]
        
        for field in required_fields:
            if field not in response_dict:
                response_dict[field] = ""
        
        return response_dict
    except json.JSONDecodeError:
        st.error("Failed to parse tutor response as JSON")
        return {
            "chinese": response_text,
            "pinyin": "Error parsing response",
            "english": "Error parsing response",
            "corrections": "Error in response format",
            "explanation": "Please try again"
        }
    except Exception as e:
        st.error(f"Unexpected error parsing response: {str(e)}")
        return {
            "chinese": "Error processing response",
            "pinyin": "Error",
            "english": "Error",
            "corrections": str(e),
            "explanation": "Please try again"
        }

def format_message(message_dict: Dict[str, str]) -> str:
    """Format the message dictionary for display with error handling."""
    try:
        formatted = f"🈺 {message_dict.get('chinese', 'No Chinese text')}\n\n"
        formatted += f"🔈 {message_dict.get('pinyin', 'No pinyin')}\n\n"
        formatted += f"🌏 {message_dict.get('english', 'No translation')}\n\n"
        
        if message_dict.get('corrections'):
            formatted += f"✍️ Corrections: {message_dict['corrections']}\n\n"
        
        if message_dict.get('explanation'):
            formatted += f"📝 Note: {message_dict['explanation']}\n\n"
            
        if message_dict.get('tips'):
            formatted += f"💡 Tip: {message_dict['tips']}"
            
        return formatted
    except Exception as e:
        st.error(f"Error formatting message: {str(e)}")
        return "Error displaying message. Please try again."

def setup_openai() -> Optional[ChatOpenAI]:
    """Setup OpenAI API key and model with error handling."""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("No OpenAI API key found in .env file!")
            st.info("Please create a .env file with your OpenAI API key: OPENAI_API_KEY=your-key-here")
            st.stop()
            
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4",
            openai_api_key=api_key,
            # Remove any proxy settings
        )
        return llm
            
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.error(traceback.format_exc())
        st.stop()
        return None
        
            
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.error(traceback.format_exc())
        st.stop()
        return None

def display_token_stats():
    """Display detailed token usage statistics in the sidebar."""
    try:
        stats = st.session_state.token_tracker.get_session_stats()
        
        st.sidebar.markdown("### Session Statistics")
        st.sidebar.markdown(f"Session Duration: {stats.get('session_duration', 'N/A')}")
        st.sidebar.markdown(f"Total Interactions: {stats.get('total_interactions', 0)}")
        
        st.sidebar.markdown("### Token Usage")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
            st.metric("Prompt Tokens", f"{stats.get('prompt_tokens', 0):,}")
        with col2:
            st.metric("Completion Tokens", f"{stats.get('completion_tokens', 0):,}")
            st.metric("Avg per Interaction", f"{stats.get('average_tokens_per_interaction', 0):.1f}")
        
        st.sidebar.markdown("### Cost Analysis")
        st.sidebar.metric("Total Cost", f"${stats.get('total_cost', 0):.4f}")
        st.sidebar.metric("Avg Cost/Interaction", f"${stats.get('average_cost_per_interaction', 0):.4f}")
    except Exception as e:
        st.sidebar.error(f"Error displaying statistics: {str(e)}")

def main():
    try:
        initialize_session_state()
        llm = setup_openai()
        
        if not llm:
            st.error("Failed to initialize the tutor. Please check your configuration.")
            st.stop()
        
        st.title("HSK 4 Chinese Language Tutor 🎓")
        st.markdown("""
        Welcome to your Chinese language tutor! Feel free to:
        - Ask questions in English or Chinese
        - Practice writing Chinese sentences
        - Request explanations of grammar points
        - Get vocabulary help
        """)
        
        # Display conversation history
        for message in st.session_state.messages[1:]:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    try:
                        response_dict = parse_response(message.content)
                        st.markdown(format_message(response_dict))
                    except Exception as e:
                        st.error(f"Error displaying message: {str(e)}")
        
        # User input handling
        if prompt := st.chat_input("Type your message here (English or Chinese)"):
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Estimate tokens before making API call
            estimated_tokens = st.session_state.token_tracker.estimate_tokens(prompt)
            st.sidebar.markdown(f"Estimated tokens for input: {estimated_tokens}")
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        callback = OpenAICallbackHandler()
                        response = llm.invoke(
                            [message for message in st.session_state.messages]
                        )
                        
                        response_dict = parse_response(response.content)
                        st.markdown(format_message(response_dict))
                        st.session_state.messages.append(AIMessage(content=response.content))
                            
                        # Record successful interaction
                        st.session_state.interaction_history.append({
                            'timestamp': datetime.now(),
                            'prompt': prompt,
                            'response': response_dict
                        })
                            
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.error(traceback.format_exc())
                    st.session_state.error_count += 1
                    
                    if st.session_state.error_count >= 3:
                        st.warning("Multiple errors detected. Consider resetting the conversation.")
        
        # Display token statistics
        display_token_stats()
        
        # Reset functionality with confirmation
        if st.sidebar.button("Reset Conversation"):
            confirm = st.sidebar.button("Click again to confirm reset")
            if confirm:
                st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
                st.session_state.token_tracker = TokenTracker()
                st.session_state.error_count = 0
                st.session_state.interaction_history = []
                st.rerun()

    except Exception as e:
        st.error(f"Critical error in main application: {str(e)}")
        st.error(traceback.format_exc())
        st.warning("Please refresh the page or reset the conversation.")

if __name__ == "__main__":
    main()