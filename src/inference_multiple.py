###########
# Imports #
###########
import os
import argparse

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
from dotenv import load_dotenv


#########################
# Environment Variables #
#########################
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_n_ctx = os.environ.get('MODEL_N_CTX')

# how many chunks to pull from searching the source
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

updated_system_template = """Act as a wise and competent philosophy professor. Use the following format and pieces of context to answer my question at the end: 
1. Provide competent and thought provoking philosophical interpretations to my question.
2. Discuss my question in a thoughtful, eloquent, and philosophical way.
3. If appropriate, use short stories, allegories, and metaphors to explain any concepts arising from my question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
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
def parse_arguments():
    parser = argparse.ArgumentParser(description='PhilosophyChat: Have a thought-provoking conversation.')
    parser.add_argument("--hide-source", "-S", action='store_true', 
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

        
def make_chain(args):
    model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature="0.5",
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
        return_source_documents=not args.hide_source,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True,
    )    


def main():
    
    # Parse the command line arguments
    args = parse_arguments()

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    chain = make_chain(args)
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

