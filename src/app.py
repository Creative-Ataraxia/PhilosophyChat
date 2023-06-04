###########
# Imports #
###########

import os
import base64
import asyncio
import traceback
from PIL import Image

import openai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from transformers import AutoTokenizer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# import modules
from config import *


#############
# Functions #
#############

@st.cache_data(show_spinner=False)
def get_local_img(file_path: str) -> str:
    # Load a byte image and return its base64 encoded string
    return base64.b64encode(open(file_path, "rb").read()).decode("utf-8")

@st.cache_data(show_spinner=False)
def get_favicon(file_path: str):
    # Load a byte image and return its favicon
    return Image.open(file_path)

@st.cache_data(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2", low_cpu_mem_usage=True)

@st.cache_data(show_spinner=False)
def get_css() -> str:
    # Read CSS code from style.css file
    with open(os.path.join(ROOT_DIR, "src", "style.css"), "r") as f:
        return f"<style>{f.read()}</style>"

def get_chat_message(contents: str = "", align: str = "left") -> str:
    # Formats the message in an chat fashion (user right, reply left)
    div_class = "AI-line"
    color = "rgb(240, 242, 246)"
    file_path = os.path.join(ROOT_DIR, "src", "assets", "AI_icon.png")
    src = f"data:image/gif;base64,{get_local_img(file_path)}"
    if align == "right":
        div_class = "human-line"
        color = "rgb(91, 133, 69)"
        if "USER" in st.session_state:
            src = st.session_state.USER.avatar_url
        else:
            file_path = os.path.join(ROOT_DIR, "src", "assets", "user_icon.png")
            src = f"data:image/gif;base64,{get_local_img(file_path)}"
    icon_code = f"<img class='chat-icon' src='{src}' width=32 height=32 alt='avatar'>"
    formatted_contents = f"""
    <div class="{div_class}">
        {icon_code}
        <div class="chat-bubble" style="background: {color};">
        &#8203;{contents}
        </div>
    </div>
    """
    return formatted_contents

class StreamingResponseAccumulator(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.data = []
        self.placeholder = st.empty()  # Create an empty text area
        self.message = ""

    def _accumulate(self, token: str, **kwargs) -> None:
        # self.data.append(token)
        self.message += token
        self.placeholder.markdown(get_chat_message(self.message), unsafe_allow_html=True)
        # self.placeholder.write(f"Wise Old Philosopher: {self.message}")  # Update the text area with the new message


    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._accumulate(token)

def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0.7",
        streaming=True,
    )
    
    embedding = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    vector_store = Chroma(
        collection_name="May-2023-Philosophy",
        embedding_function=embedding,
        persist_directory=persist_directory,
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(search_kwargs={"k": target_source_chunks}),
        return_source_documents=SHOW_SOURCE_DOCUMENTS,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )    


async def main(question: str) -> dict:
    res = {'status': 0, 'message': "Success"}
    chain = make_chain()
    accumulator = StreamingResponseAccumulator()
    chat_history = [] 

    try:
        # Strip the prompt of any potentially harmful html/js injections
        question = question.replace("<", "&lt;").replace(">", "&gt;")

        # Update both chat log and the model memory
        st.session_state.Log.append(f"Seeker: {question}")
        st.session_state.Memory.append({'role': "user", 'content': question})

        # Clear the input box after question is submitted
        prompt_box.empty()

        with chat_box:
            # Write the latest human message first
            line = st.session_state.Log[-1]
            contents = line.split("Seeker: ")[1]
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

            # reply_box = st.empty()
            # reply_box.markdown(get_chat_message(), unsafe_allow_html=True)

            # This is one of those small three-dot animations to indicate the bot is "writing"
            writing_animation = st.empty()
            file_path = os.path.join(ROOT_DIR, "src", "assets", "loading.gif")
            writing_animation.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<img src='data:image/gif;base64,{get_local_img(file_path)}' width=30 height=10>", unsafe_allow_html=True)

            # Get answer from the chain
            response = chain({"question": question, "chat_history": chat_history}, callbacks=[accumulator])
            answer, source = response["answer"], [] if not SHOW_SOURCE_DOCUMENTS else response['source_documents']

            if DEBUG:
                with st.sidebar:
                    st.write("openai_api_response:")
                    st.json({'str': response}, expanded=False)
                    st.json(st.session_state.Memory, expanded=False)

            # Render the reply as chat reply
            # reply_box.markdown(get_chat_message(answer), unsafe_allow_html=True)

            # Clear the writing animation
            writing_animation.empty()

            # Update the chat log and the model memory
            st.session_state.Log.append(f"Philosopher: {answer}")
            st.session_state.Memory.append({'role': "assistant", 'content': answer})

    except:
        st.write(traceback.format_exc())

    return res


#####################
# Environment setup #
#####################

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

errors = []

# Set your OpenAI API Key; streamlit's doc: https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management
openai.api_key = st.secrets["OPENAI_API_KEY"]

embeddings_model_name = EMBEDDINGS_MODEL_NAME
persist_directory = PERSIST_DIRECTORY
model_n_ctx = MODEL_N_CTX

# how many chunks to pull from searching the source
target_source_chunks = int(TARGET_SOURCE_CHUNKS)

updated_system_template = """Act as a wise and competent philosophy professor. Use the following format and pieces of context to answer my question at the end: 
1. Provide competent and thought provoking philosophical interpretations to my question.
2. Discuss my question in a thoughtful, eloquent, and philosophical way.
3. If appropriate, use short stories, allegories, and metaphors to explain any concepts arising from my question.
Always use the tone of an old wise sage. Never break character. Try to give source of the knowledge if possible. Always respond in the same language as the question.
----------------------
contexts:
{context}"""

updated_messages = [
    SystemMessagePromptTemplate.from_template(updated_system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

qa_prompt = ChatPromptTemplate.from_messages(updated_messages)

if len(errors) > 0:
    st.error("\n".join(errors))
    st.stop()


######################
# Initialize the App #
######################

# Icons
favicon = get_favicon(os.path.join(ROOT_DIR, "src", "assets", "AI_icon.png"))

# Page layout settings
st.set_page_config(
    page_title="Philosophy Chat",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="collapsed" if DEBUG else "auto",
    menu_items={
        'Get help': 'https://twitter.com/',
        'Report a bug': "https://github.com/Creative-Ataraxia",
        'About': "This conversational AI is embedded with 500MBs of philosophy texts, and should be able to engage in in-depth philosophical discussions."
    }
)

# Get options from URL
query_params = st.experimental_get_query_params()
if "debug" in query_params and query_params["debug"][0].lower() == "true":
    st.session_state.DEBUG = True

if "DEBUG" in st.session_state and st.session_state.DEBUG:
    DEBUG = True

# Init Tokenizer
with st.spinner("Loading..."):
    TOKENIZER = get_tokenizer()  # First time after deployment takes a few seconds


################
# Streamlit UI #
################

# Define main layout
st.title("Welcome")
st.subheader("to Philosophy Chat!")
st.subheader("")
chat_box = st.container()
add_vertical_space(2)
prompt_box = st.empty()
add_vertical_space(2)  

if DEBUG:
    with st.sidebar:
        st.subheader("Debug area")

st.markdown(get_css(), unsafe_allow_html=True)


####################
# State Management #
####################

if "Memory" not in st.session_state:
    st.session_state.Memory = []
    st.session_state.Log = []

# Render chat history so far
with chat_box:
    for line in st.session_state.Log[1:]:
        # For AI response
        if line.startswith("Philosopher: "):
            contents = line.split("Philosopher: ")[1]
            st.markdown(get_chat_message(contents), unsafe_allow_html=True)

        # For human prompts
        if line.startswith("Seeker: "):
            contents = line.split("Seeker: ")[1]
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

# Define an input box for human prompts
with prompt_box:
    question = st.text_input("Your Philosophical Inquiry:", value="", help="Ask any philosophical questions", key=f"text_input_{len(st.session_state.Log)}")

# Gate the subsequent chatbot response to only when the user has entered a prompt
if len(question) > 0:
    run_res = asyncio.run(main(question))

    # if main() return with success status:
    if run_res['status'] == 0 and not DEBUG:
        # rerun to react to updates
        st.experimental_rerun()
    else:
        if run_res['status'] != 0:
            st.error(run_res['message'])
        with prompt_box:
            if st.button("Show Text Box"):
                st.experimental_rerun()

st.write('Made with ❤️ by [Creative_Ataraxia](<https://github.com/Creative-Ataraxia?tab=repositories>)')