# If necessary, install the openai Python library by running 
# pip install openai


from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core import StorageContext, load_index_from_storage
from bs4 import BeautifulSoup
from urllib import request
import os
import kdbai_client as kdbai
import time
import pandas as pd


def get_response2(k, user_query, chat_history, base_url, system_msg, max_tokens):

    #Set up KDB.AI endpoint and API key
    KDBAI_ENDPOINT = (
        os.environ["KDBAI_ENDPOINT"]
    )
    KDBAI_API_KEY = (
        os.environ["KDBAI_API_KEY"]
    )

    #connect to KDB.AI
    session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)
    db = session.database('myDatabase')
    table = db.table('multi_modal_ImageBind')
    #Create a query vector for similarity search
    query_vector = [get_hf_embedding(user_query)]
    results = mm_search(table,query_vector, k) 

    template = """
    {system_msg} Answer the following questions considering the history of the conversation:
    Chat history: {chat_history}
    User question: {user_question}
    Use the following documents when answering questions:
    Documents: {documents}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        base_url=base_url,
        max_tokens=max_tokens,
        api_key="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" 
	)         
    chain = prompt | llm | StrOutputParser()
 
    image_links = ''
    
    documents = results[0]
   
    doc_texts =  "\\n".join( doc for doc in documents[documents['media_type']=='text']['text']) 
    image_links = "\\n".join( doc for doc in documents[documents['media_type']=='image']['path']) 
    citation_links = "\\n".join( doc for doc in documents[documents['media_type']=='text']['path']) 
               
    return chain.stream({
        "system_msg" :system_msg,
        "chat_history": chat_history,
        "user_question": user_query,
        "documents": doc_texts,
    }), image_links, citation_links


# Multimodal search function, identifies most relevant images and text within the vector database
def mm_search(table, query, k=1):
    #image_results = table.search(vectors={"flat_index":query}, n=k, filter=[("like", "media_type", "image")])
    text_results = table.search(vectors={"flat_index":query}, n=k, filter=[("like", "media_type", "text")])
    text_results_ids = text_results[0]['element_id']
        
    text_results_ids_str = []
    for element_id in text_results_ids:
        text_results_ids_str.append(str(element_id))
        
    image_results = table.query( filter=[("like", "media_type", "image"),("in", "element_parent_id", text_results_ids_str) ])
      
    if len(image_results) > 0:
        results = [pd.concat([text_results[0], image_results ], ignore_index=True)]
    else:
        results = [pd.concat([text_results[0]], ignore_index=True)]
      
    return(results)

from langchain_huggingface import HuggingFaceEndpointEmbeddings
import numpy as np
from retrying import retry

@retry(wait_fixed=3000,stop_max_attempt_number=3)
def get_hf_embedding(text_data):
    model = "sentence-transformers/all-mpnet-base-v2"
    #model = "sentence-transformers/all-roberta-large-v1"
          
    hf = HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token="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"
    )
    embedding =hf.embed_query(text_data)
    vector = np.array(embedding)
    return(vector)


   
       
       
        
