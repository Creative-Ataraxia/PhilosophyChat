# Debug switch
DEBUG = False

# Configs
SHOW_SOURCE_DOCUMENTS = False

# Generic internet settings
TIMEOUT = 60
N_RETRIES = 3
COOLDOWN = 2
BACKOFF = 1.5

# Settings for OpenAI NLP models. Here, NLP tokens are not to be confused with user chat or image generation tokens
NLP_MODEL_NAME = "gpt-3.5-turbo"                   
NLP_MODEL_MAX_TOKENS = 4000
NLP_MODEL_REPLY_MAX_TOKENS = 1500
NLP_MODEL_TEMPERATURE = 0.5
NLP_MODEL_FREQUENCY_PENALTY = 1
NLP_MODEL_PRESENCE_PENALTY = 1
NLP_MODEL_STOP_WORDS = ["Human:", "AI:"]

# Environment Variables
PERSIST_DIRECTORY="data/chroma"
SOURCE_DIRECTORY="data/encyclopedia"
EMBEDDINGS_MODEL_NAME="all-MiniLM-L6-v2"
MODEL_N_CTX=4000

CHUNK_SIZE=1500
CHUNK_OVERLAP=150
TARGET_SOURCE_CHUNKS=4

INITIAL_PROMPT="Welcome to Philosophy Chat"