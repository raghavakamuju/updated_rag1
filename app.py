import os
from langchain import HuggingFaceHub
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import time
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
import pickle
import streamlit as st
website_links = [
 'https://gould.usc.edu/academics/degrees/online-llm/',
 'https://gould.usc.edu/academics/degrees/llm-1-year/application/',
 'https://gould.usc.edu/academics/degrees/two-year-llm/application/',
 'https://gould.usc.edu/academics/degrees/llm-in-adr/application/',
 'https://gould.usc.edu/academics/degrees/llm-in-ibel/application/',
 'https://gould.usc.edu/academics/degrees/llm-in-plcs/application/',
 'https://gould.usc.edu/academics/degrees/online-llm/application/',
 'https://gould.usc.edu/academics/degrees/mcl/',
 'https://gould.usc.edu/academics/degrees/mcl/application/',
 'https://gould.usc.edu/news/what-can-you-do-with-an-llm-degree/',
 'https://gould.usc.edu/news/msl-vs-llm-vs-jd-which-law-degree-is-best-for-your-career-path/',
 'https://gould.usc.edu/news/three-things-ll-m-grads-wish-they-knew-when-they-started/',
  'https://gould.usc.edu/academics/degrees/llm-1-year/', 
  'https://gould.usc.edu/academics/degrees/two-year-llm/',
  'https://gould.usc.edu/academics/degrees/llm-in-adr/',
  'https://gould.usc.edu/academics/degrees/llm-in-ibel/',
  'https://gould.usc.edu/academics/degrees/llm-in-plcs/'
]
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
load_dotenv()
open_ai_token = st.secrets('api_key')
if os.path.exists("chunks_embeddings.pkl"):
    with open("chunks_embeddings.pkl", "rb") as f:
        chunks, chunk_embeddings = pickle.load(f)
else:
    texts = []
    for link in website_links:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        texts.append(text)
    combined_text = ",".join(texts)
    splitter = CharacterTextSplitter(chunk_size=1000)
    chunks = splitter.split_text(combined_text)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    chunk_embeddings = [model.encode(chunk) for chunk in chunks]

    with open("chunks_embeddings.pkl", "wb") as f:
        pickle.dump((chunks, chunk_embeddings), f)

dimension = chunk_embeddings[0].shape[0]
faiss_index = faiss.IndexFlatL2(dimension)
embeddings_array = np.array(chunk_embeddings).astype('float32')
faiss_index.add(embeddings_array)
from langchain.llms import OpenAI
llm_openai = OpenAI(temperature=0.6,openai_api_key=open_ai_token)

def answer(text, model,listing_history):
    if len(listing_history)>=5:
      listing_history=listing_history[len(listing_history)-5:]

    prompt_template = PromptTemplate(
        input_variables=['query_text', 'retrieved','listing_history'],
        template="Given the following information: '{retrieved}' and {listing_history} . Answer the question: '{query_text}'. "
    )
    start_time = time.time()
    query_embedding = model.encode(text)
    query_embedding = np.array([query_embedding]).astype('float32')
    k = 10
    D, I = faiss_index.search(query_embedding, k)
    retrieved_list = [chunks[i] for i in I[0]]
    end_time = time.time()
    chain = LLMChain(llm=llm_openai, prompt=prompt_template)
    response_stream = chain.run(query_text=text, retrieved=retrieved_list,listing_history=listing_history,stream=True)
    return response_stream, end_time - start_time
listing_history=[]
import streamlit as st

st.header("Welcome to LLM Chatbot")

if 'listing_history' not in st.session_state:
    st.session_state['listing_history'] = []

user_query = st.text_input("Enter the query:", key="user_query")

if st.button("Answer"):
    if user_query.lower() == "exit":
        st.stop() 
    else:
        output, time_elapsed = answer(user_query, model, st.session_state['listing_history'])
        st.session_state['listing_history'].append((user_query, output)) # Store both question and answer
        st.write(user_query)
        st.write(output) 

st.write("Previous Chats:")
for question, answer in st.session_state['listing_history']:
    st.write(f"Question: {question}")
    st.write(f"Answer: {answer}")
