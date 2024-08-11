import os

from dotenv import load_dotenv
import dspy
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dspy.retrieve.chromadb_rm import ChromadbRM
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import manage_vector_db as db_util

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL")


def load_dataset():
    """
    Create list of dspy.primitives.example.Example from the `dune_questions.txt`.
    """
    with open('./dune_questions.txt', 'r') as f:
       lines = f.readlines()
    data = []
    for line in lines:
       q_and_a = line.replace('\n', '').split(';')
       ex = dspy.Example(question=q_and_a[0], answer=q_and_a[1]).with_inputs("question")
       data.append(ex)

    return data

def configure_dspy_v1():
    """
    This configuration uses OpenAI gpt-mini for the LLM model
    and a Chroma vector database as the retriever model. 
    """
    llm = dspy.OpenAI(model='gpt-4o-mini')

    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name="text-embedding-ada-002"
    )

    retriever_model = ChromadbRM(
        'Dune',
        os.environ["VECTOR_DB_DIR"],
        embedding_function=embedding_function,
        k=5
    )

    dspy.settings.configure(lm=llm, rm=retriever_model)
    
    return llm, retriever_model

def get_openai_few_shot_prompt():

    template = """The following are multiple choice questions with answers. Return the letter of the correct answer.

    Question: Who is the first President of the United States?
    - A. Herbert Hoover
    - B. George Washington
    - C. Bugs Bunny
    - D. Joe Biden
    Correct Answer: B

    Question: When wrote the 1977 book The Shinning?
    - A. Stephen King
    - B. Kathy Bates
    - C. Joe Hill
    - D. Jack Nicholson
    Correct Answer: A

    Question: What is the main ingredient of household dust?
    - A. Air pollution
    - B. Rocks
    - C. Dead skin cell
    - D. Insects
    Correct Answer: C

    {question} 
    """

    summary_prompt = PromptTemplate.from_template(template=template)

    llm = ChatOpenAI(temperature=0, model_name=os.getenv("OPENAI_LLM_MODEL")) 
    chain = summary_prompt | llm | StrOutputParser()

    return chain