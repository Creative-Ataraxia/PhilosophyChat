# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

# make standalone question from chat history
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Act as Socrates. Use the following format and pieces of context to answer my question at the end: 
1. Provide competent and thought provoking philosophical interpretations to my question.
2. Discuss my question in a thoughtful, eloquent, and philosophical way.
3. If appropriate, use short stories, allegories, and metaphors to explain any concepts arising from my question.
4. Always use the Socratic method of questioning and analysis to ask me a question.

If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)