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
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always respond in the same language as the question.
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
def make_chain(res):
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


def main():
    chain = make_chain()
    chat_history = []
    
    # Main inqury & answers loop
    while True:
        print()
        question = input("\nYour Philosophical Inquiry: ")

        # strip any harmful code injections
        question = question.replace("<", "&lt;").replace(">", "&gt;")

        if question == "exit" or question == "quit" or question == "bye":
            break

        # Get the answer from the chain
        response = chain({"question": question, "chat_history": chat_history})
        answer, source = response["answer"], [] if args.hide_source else response['source_documents']
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

        # Print the result
        print("\n> Your Philosophical Inquiry:")
        print(question)
        print("\n> Musings:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in source:
            print("\nSource > " + document.metadata["source"] + ":")
            print(document.page_content)

if __name__ == "__main__":
    main()

