import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from utils import get_chat_message


class StreamingResponseAccumulator(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.data = []
        self.placeholder = st.empty()  # Create an empty text area
        self.message = ""

    def _accumulate(self, token: str, **kwargs) -> None:
        self.message += token
        self.placeholder.markdown(get_chat_message(self.message), unsafe_allow_html=True)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._accumulate(token)