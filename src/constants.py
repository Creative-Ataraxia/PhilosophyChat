from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


# Configs
DEBUG = False
SHOW_SOURCE_DOCUMENTS = False

# Inferencing Environment Variables
PERSIST_DIRECTORY="data/chroma"
MODEL_N_CTX=4000
TARGET_SOURCE_CHUNKS=4

# Embedding Environment Variables
CHUNK_SIZE=1500
CHUNK_OVERLAP=150
SOURCE_DIRECTORY="data/encyclopedia"
EMBEDDINGS_MODEL_NAME="all-MiniLM-L6-v2"


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

QA_PROMPT = ChatPromptTemplate.from_messages(updated_messages)

INITIAL_MESSAGE = "Philosopher: Welcome Seeker. Which philosophical topics would you like to discuss today?"