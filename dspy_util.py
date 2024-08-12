"""
Utility script for DSPy APIs.
"""
import json
import os
import random

import dspy
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from dspy.evaluate.evaluate import Evaluate
from dspy.retrieve.chromadb_rm import ChromadbRM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import nltk
nltk.download('punkt_tab')

import manage_vector_db as db_util

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

def load_dataset():
    """
    Create list of dspy.Example from the `dune_questions.json`.
    """

    with open("dune_questions.json", "r") as f:
        json_data = json.load(f)
    idx_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    examples = [ dspy.Example(question=ex['question'], 
                choices='\n'.join([f"- {idx_map[idx]}. {opt}" for idx, opt in enumerate(ex['choices'].split('\n'))]),
                answer=ex['truth']).with_inputs("question", "choices")  for ex in json_data['questions']]
   
    return examples

def configure_dspy_v1():
    """
    This configuration uses OpenAI gpt-mini for the LLM model
      and a Chroma vector database as the retriever model. 

    NOTE: The code below assume a Chroma vector database has been 
      created in the directory VECTOR_DB_DIR. If this vector DB has not been created,
      call the manage_vector_db.create_vector_db() method. It will create the 
      vector database in the VECTOR_DB_DIR directory.
    """
    llm = dspy.OpenAI(model=LLM_MODEL)

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
    """
    Create an OpenAI LLM using a few-shot multi-choice prompt.
    Return OpenAI LLM model.
    """

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

def llm_metric(gold, pred, trace=None):
    """
    Determine if truth value passed is equal to prediction.
    Parameters:
    - gold (dspy.Example): Truth value for evaluation.
    - pred (dspy.Prediction): Model prediction.
    Return 1 when truth value equals prediction, 0 otherwise.
    """
    #print(f'INPUT: {gold.question} | {gold.answer} | {pred.answer} | {gold.choices.replace("\n",",")}')
    correct = gold.answer.strip().lower() == pred.answer.strip().lower() 
    score = 1 if correct else 0
    return score

def evaluate(dev, model):
    """
    Using the `llm_metric` function, evalute the llm model.
    Parameters:
    - dev (list): A list of dspy.Example instances.
    - model (LLM model): DSpy model instance.
    Return DSPy evaluation results.
    """
    
    # Set up the evaluate function. 
    evaluate_model = Evaluate(devset=dev, num_threads=1, display_progress=True, display_table=False, return_outputs=True)

    metric = evaluate_model(model, metric=llm_metric)

    return metric