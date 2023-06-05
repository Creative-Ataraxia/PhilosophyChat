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
Always use the tone of an old wise sage. Never break character. If there are specific sources in the context given, make sure to cite those sources in your answer. Always respond in the same language as the question.
----------------------
contexts:
{context}"""

updated_messages = [
    SystemMessagePromptTemplate.from_template(updated_system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

QA_PROMPT = ChatPromptTemplate.from_messages(updated_messages)

INITIAL_MESSAGE = "Philosopher: Welcome Seeker. Which philosophical topics would you like to discuss today?"

PRE_MADE_PROMPTS = [
  "What is consciousness and can it be definitively measured or observed?",
  "Does the concept of free will hold true in a deterministic universe governed by the laws of physics?",
  "Is there an objective reality, or is all reality subjective and created by our perceptions and experiences?",
  "Is there inherent meaning or purpose to life, or is it our responsibility to create our own purpose?",
  "Is it possible to have a universal morality, or are ethical norms purely a product of culture and personal beliefs?",
  "Can we ever truly understand another individual's subjective experience, or are we eternally confined to our own perspective?",
  "Does our identity reside entirely in the physical brain, or is there something more to 'self'?",
  "How can we distinguish human consciousness from potential artificial intelligence consciousness?",
  "Is the notion of 'I' an illusion, a construct of the brain, or is there a persistent self?",
  "If time is infinite or cyclical, what significance does our existence hold?",
  "What is the nature of beauty and why are we attracted to it?",
  "Is it morally acceptable to sacrifice one for the greater good of many?",
  "If we can extend our lives indefinitely through technology, should we do so?",
  "What is the role of suffering in human life and could it be seen as necessary?",
  "What does it mean to lead a good life and is this universal or individual?",
  "Can we truly possess knowledge, or are we constantly in a state of belief and revision?",
  "Does our existence precede essence, or does our essence precede our existence?",
  "What is the nature of death and why is it universally feared or avoided?",
  "Can there be a 'just' war and if so, under what circumstances?",
  "What is the true nature of love, is it a feeling, a decision, or something else?",
  "What does it mean for something to be 'natural', and why do we value it?",
  "Are we morally obligated to future generations, and if so, to what extent?",
  "Is there a difference between the brain and the mind, or are they the same?",
  "What role do dreams play in our consciousness and understanding of reality?",
  "What is the nature of truth, and can it ever be known absolutely?"
]