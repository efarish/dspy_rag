"""
A script for creating a vector data store using Chroma vector db.

Entry point is the function create_vector_db function.
"""
import logging
import os
import re
from typing import List 

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from nltk.tokenize import word_tokenize

load_dotenv()

VECTOR_DB_DIR = os.environ["VECTOR_DB_DIR"]

def create_docs(list_of_text: List[str]) -> List[Document]:
    """
    Utility function to create a list of LangChain documents 
      from a list of strings. 
    """
    docs = []
    for idx, chapter in enumerate(list_of_text):
        docs.append(Document(page_content=chapter, metadata={"split": idx+1}))
    return docs


def get_Dune_chapters() -> List[Document]: 
    """
    Extract the chapters form the source corpus.
    Returns: LangChain Document objects.
    """
    with open('dune.txt', 'r', encoding="utf-8") as f:
        text = f.read()
    # For this file, "= = = = = =" separates the chapters.
    text_split = text.split('= = = = = =')
    # Exclude appendicies and splits with less than 10 words.
    cleaned_text = []
    for idx, line in enumerate(text_split):
        if not re.search("Appendix .+:", line) and \
        not re.search('Terminology of the Imperium', line) and \
        len( word_tokenize(line.strip()) ) > 10:
            trimmed_line = line.strip()
            cleaned_text.append( trimmed_line )
    # This text file contains spaces before each paragraph extra line feeds.
    #  The code below cleans that up.
    for idx, chapter in enumerate(cleaned_text):
        cleaned_text[idx] = chapter.replace('\n    ', '\n').replace('\n\n','\n')
    # Create LangChain Document instances.
    lc_docs = create_docs(cleaned_text)
    return lc_docs

def load_data(number_of_splits=5, split_size=500) -> List[Document]:
    """
    TODO
    
    Args:
        None.

    Returns:
        str: Two lists. One contains the text splits, the other is the split meta data.
    """

    dune_chapters = get_Dune_chapters()
    splitter = TokenTextSplitter(encoding_name='gpt2', chunk_size=split_size, chunk_overlap=50)
    dune_splits = splitter.split_documents(dune_chapters[:number_of_splits])
 
    return dune_splits

def create_vector_db() -> Chroma:
    """
    Create a vector data store if one does not exist.
    If the vector store has already been created, load it and return it.
    Args:
        None.

    Returns:
        str:  Vector data store instance.
    """

    embeddings = OpenAIEmbeddings()
    if not os.path.exists(VECTOR_DB_DIR):
        logging.warning("Source directory doesn't exists. Create vector data store...")
        embeddings = OpenAIEmbeddings()
        texts = load_data()      
        db = Chroma.from_documents(persist_directory=VECTOR_DB_DIR, documents=texts, embedding=embeddings)        
        logging.info("Done.")
    else:
        logging.info("Data source exists. Loading vector data store.")
        db = Chroma(persist_directory=VECTOR_DB_DIR,embedding_function=embeddings)

    return db

def get_retriever(k: int=5) -> VectorStoreRetriever:
    """
    Gets a Chroma retriever.
    Args: k - the number of documents to retrieve.
    Returns: VectorStoreRetriever instance.
    """
    db = create_vector_db()
    return db.as_retriever(search_kwargs={"k": k})


