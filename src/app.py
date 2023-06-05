###########
# Imports #
###########
# System
import os
import asyncio
import traceback
import random

# Libraries
import openai
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Modules
from constants import *
from utils import *
from callback_override import StreamingResponseAccumulator


################
# Environments #
################
# Set your OpenAI API Key; streamlit's doc: https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management
openai.api_key = st.secrets["OPENAI_API_KEY"]


def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0.7",
        streaming=True,
    )
    
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    vector_store = Chroma(
        collection_name="May-2023-Philosophy",
        embedding_function=embedding,
        persist_directory=PERSIST_DIRECTORY,
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS}),
        return_source_documents=SHOW_SOURCE_DOCUMENTS,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=False,
    )    


def select_premade_prompt():
    i = random.randint(0, len(PRE_MADE_PROMPTS)-1)
    question = PRE_MADE_PROMPTS[i]
    return question


async def main(question: str):
    llmchain = make_chain()
    accumulator = StreamingResponseAccumulator()
    chat_history = [] 
    res = {'status': 0, 'message': "Success"}

    try:
        # Strip the prompt of any potentially harmful html/js injections
        question = question.replace("<", "&lt;").replace(">", "&gt;")

        # Update memory with question
        st.session_state.Log.append(f"Seeker: {question}")
        st.session_state.Memory.append({'role': "user", 'content': question})

        # Clear the input box after question is submitted
        prompt_box.empty()

        with chat_box:
            # Write the latest human message first
            line = st.session_state.Log[-1]
            contents = line.split("Seeker: ")[1]
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

            # Get answer from the chain
            response = llmchain({"question": question, "chat_history": chat_history}, callbacks=[accumulator])
            # render source documents later if needed
            answer, source = response["answer"], [] if not SHOW_SOURCE_DOCUMENTS else response['source_documents']

            if DEBUG:
                with st.sidebar:
                    st.write("openai_api_response:")
                    st.json({'str': response}, expanded=False)
                    st.json(st.session_state.Memory, expanded=False)

            # Update memory with answer
            st.session_state.Log.append(f"Philosopher: {answer}")
            st.session_state.Memory.append({'role': "assistant", 'content': answer})

    except:
        res['status'] = 2
        res['message'] = traceback.format_exc()
    
    return res


######################
# Initialize the App #
######################
st.set_page_config(
    page_title="Philosophy Chat",
    page_icon=get_asset(os.path.join(ROOT_DIR, "src", "assets", "AI_icon.png"), is_image=True),
    layout="wide",
    initial_sidebar_state="collapsed" if DEBUG else "auto",
    menu_items={
        'Get help': 'https://github.com/Creative-Ataraxia',
        'Report a bug': "https://github.com/Creative-Ataraxia",
        'About': "This conversational AI is embedded with 500MBs of philosophy texts. Given the right prompt layouts, this Chatbot should be able to engage in in-depth philosophical discussions."
    })

st.markdown(get_css(), unsafe_allow_html=True)

# Get options from URL
query_params = st.experimental_get_query_params()
if "debug" in query_params and query_params["debug"][0].lower() == "true":
    st.session_state.DEBUG = True

# Check debug flag
if "DEBUG" in st.session_state and st.session_state.DEBUG:
    DEBUG = True

if DEBUG:
    with st.sidebar:
        st.subheader("Debug area")

# Init Tokenizer
with st.spinner("Loading tokenizer..."):
    TOKENIZER = get_tokenizer()  # First time deployment takes a few seconds


################
# Streamlit UI #
################
# State Management
if "Memory" not in st.session_state:
    st.session_state.Memory = []
    st.session_state.Log = [INITIAL_MESSAGE]

# Define main layout
st.title("Welcome to Philosophy Chat / 哲学畅谈 / 哲学チャット!")
st.divider()
st.markdown("##### Chat with over 250,000 pages of modern & historical philosophy texts")
st.markdown("##### 与超过25万页古典和现代哲学典籍对话")
st.markdown("##### 25万ページ以上の現代・歴史哲学のテキストを使ったチャット")
add_vertical_space(2)
chat_box = st.container()
st.divider()
prompt_box = st.empty()
premade_prompt_container = st.empty()
add_vertical_space(1)


##############
# Main Logic #
##############
# Render chat history so far
with chat_box:
    for line in st.session_state.Log:
        # For AI response
        if line.startswith("Philosopher: "):
            contents = line.split("Philosopher: ")[1]
            st.markdown(get_chat_message(contents), unsafe_allow_html=True)

        # For human prompts
        if line.startswith("Seeker: "):
            contents = line.split("Seeker: ")[1]
            st.markdown(get_chat_message(contents, align="right"), unsafe_allow_html=True)

# Input box UI for human prompts
with prompt_box:
    question = st.text_input("Your Philosophical Inquiry:", value="", help="Explore any philosophical topics", key=f"text_input_{len(st.session_state.Log)}")

# Pre-made prompts for users
with premade_prompt_container.container():
    if st.button("You may also want to ask: "):
        Premade_Prompt = select_premade_prompt()
        st.text(Premade_Prompt)
        

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
            if st.button("Refresh the App"):
                st.experimental_rerun()


add_vertical_space(3)  
st.write('Made with ❤️ by [Creative_Ataraxia](<https://github.com/Creative-Ataraxia?tab=repositories>)')

if st.button('Empty Chat'):
    chat_box.empty()
    st.session_state.Log = [INITIAL_MESSAGE]
    st.experimental_rerun()