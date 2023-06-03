###########
# Imports #
###########
import os
import argparse

import openai
import streamlit as st
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

from config import *


#########################
# Environment Variables #
#########################

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

#############
# Functions #
#############
def make_chain():
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0.7",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
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
        verbose=True,
    )    