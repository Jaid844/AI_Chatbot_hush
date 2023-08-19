import os
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter

directory = '/content/drive/MyDrive/data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents



os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


def vector_store():
  documents=load_docs('/content/drive/MyDrive/data')
  docs=split_docs(documents)
  instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
  pinecone.init(api_key="aa24f75f-b31c-4e0c-bc0d-863ca548758a",environment="us-west4-gcp-free")
  index_name = "hush"
  index = Pinecone.from_documents(docs, instructor_embeddings, index_name=index_name)
  return index

def get_similiar_docs(query,k=2,score=False):
  index=vector_store()
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs

def get_answer(query):
  llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
  chain = load_qa_chain(llm, chain_type="stuff")
  similar_docs = get_similiar_docs(query)
  # print(similar_docs)
  answer =  chain.run(input_documents=similar_docs, question=query)
  return  answer




st.title("Question Search App")
search_query = st.text_input("Enter your question:")
if st.button("Search"):
   result = get_answer(search_query)  # Replace with your search method
   st.write("Search results:")
   result_lines = result.splitlines()
   for item in  result_lines:
        st.write(item)
