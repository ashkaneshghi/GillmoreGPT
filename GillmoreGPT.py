import streamlit as st
from langchain.vectorstores import FAISS
# from transformers import GPTNeoXForCausalLM, AutoTokenizer, pipeline
# from langchain.llms import HuggingFacePipeline
# from langchain.embeddings.base import Embeddings
# from typing import List
# from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from PIL import Image
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
import os

# class LocalHuggingFaceEmbeddings(Embeddings):
#     def __init__(self, model_id): 
#         # Should use the GPU by default
#         self.model = SentenceTransformer(model_id)
        
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Embed a list of documents using a locally running
#            Hugging Face Sentence Transformer model
#         Args:
#             texts: The list of texts to embed.
#         Returns:
#             List of embeddings, one for each text.
#         """
#         embeddings =self.model.encode(texts)
#         return embeddings

#     def embed_query(self, text: str) -> List[float]:
#         """Embed a query using a locally running HF 
#         Sentence trnsformer. 
#         Args:
#             text: The text to embed.
#         Returns:
#             Embeddings for the text.
#         """
#         embedding = self.model.encode(text)
#         return list(map(float, embedding))

# embeddings = LocalHuggingFaceEmbeddings('all-mpnet-base-v2')
# embeddings = LocalHuggingFaceEmbeddings('gtr-t5-large')
# embeddings = LocalHuggingFaceEmbeddings('gtr-t5-base')
# embeddings = LocalHuggingFaceEmbeddings('multi-qa-MiniLM-L6-cos-v1')

# tokenizer = AutoTokenizer.from_pretrained("andreaskoepf/pythia-1.4b-gpt4all-pretrain")
# base_model = GPTNeoXForCausalLM.from_pretrained(
#     "andreaskoepf/pythia-1.4b-gpt4all-pretrain",
#     load_in_8bit=True,
#     device_map='auto',
#     )

# pipe = pipeline(
#     "text-generation",
#     model=base_model,
#     tokenizer=tokenizer,
#     max_new_tokens=64,
#     temperature=0,
#     )
# llm = HuggingFacePipeline(pipeline=pipe)

os.environ['OPENAI_API_KEY'] = <>
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)

vector_store = FAISS.load_local("GillmoreGPT_index_OpenAI", embeddings)

def GillBot(q):
  input = embeddings.embed_query(q)
  docs = vector_store.similarity_search_by_vector(input, k=1)
  doc_store = FAISS.from_documents(documents=docs, embedding=embeddings)
  Gill = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type="stuff", 
                                   retriever=doc_store.as_retriever(), 
                                   return_source_documents=True,
                                   )
  res = Gill({"query": q}, return_only_outputs=True)
  st.write(res['result'].lstrip().rstrip(res['result'].split('.')[len(res['result'].split('.'))-1]).replace('\n', ' ').replace('  ', ' '))
  st.write(res['source_documents'][0].metadata['Source'])

st.set_page_config(page_title='GillmoreGPT', layout='wide')

col1, _ = st.columns([2, 1])

with col1:
    image = Image.open('wbs_primarylogo_gillmore_rgb.png')
    st.image(image, width=200)
    st.title("GillmoreGPT")

    st.markdown(" ")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("Question: ","", key="input")
    return input_text

with col1:
    question = get_text()
#    if st.button('Ask'):
    if question:
       GillBot(question)

