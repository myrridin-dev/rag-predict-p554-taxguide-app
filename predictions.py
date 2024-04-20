import json
import os
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS

import pdfminer

import langchain
from langchain import globals
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

import vertexai

PROJECT_ID='first-vertexai-project'
REGION_ID='us-central1'

TEXT_EMBEDDING_MODEL = 'textembedding-gecko'
LLM_MODEL = "gemini-1.0-pro"

INDEX_PATH = './index/'
DB_PATH = '/tmp/'

DEBUG = False

globals.set_debug(DEBUG)

vertexai.init(project=PROJECT_ID, location=REGION_ID)

def get_split_documents(index_path: str) -> List[str]:
    chunk_size=1024
    chunk_overlap=128

    split_docs = []

    for file_name in os.listdir(index_path):
        print(f"file_name : {file_name}")
        if file_name.endswith(".pdf"):
            loader = UnstructuredPDFLoader(index_path + file_name)
        else:
            loader = TextLoader(index_path + file_name)

        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs.extend(text_splitter.split_documents(loader.load()))

    return split_docs


def create_vector_db():
    embeddings = VertexAIEmbeddings(
        model_name=TEXT_EMBEDDING_MODEL, batch_size=5
    )
    # Load documents, generate vectors and store in Vector database
    split_docs = get_split_documents(INDEX_PATH)

    # Chroma
    '''
    chromadb = Chroma.from_documents(
        documents=split_docs, embedding=embeddings, persist_directory=DB_PATH
    )
    chromadb.persist()  # Ensure DB persist
    '''

    # FAISS
    faissdb = FAISS.from_documents(split_docs, embeddings)
    faissdb.save_local(DB_PATH + '/faiss.db')

    return faissdb

def get_prompt_template():
    return """
        You are a helpful AI assistant. You're tasked to answer the question given below, but only based on the context provided.

        The Tax Guide for Seniors provides a general overview of selected topics that are of interest to older tax-payers...

        Q: How do I report the amounts I set aside for my IRA?
        A: See Individual Retirement Arrangement Contributions and Deductions in chapter 3.

        Q: What are some of the credits I can claim to reduce my tax?
        A: See chapter 5 for discussions on the credit for the elderly or the disabled, the child and dependent care credit, and the earned income credit.

        Q: Must I report the sale of my home? If I had a gain, is any part of it taxable?
        A: See Sale of Home in chapter 2.

        context:
        <context>
        {context}
        </context>

        question:
        <question>
        {input}
        </question>

        If you cannot find an answer ask the user to rephrase the question.
        answer:
    """

llm = VertexAI(
    model=LLM_MODEL,
    max_output_tokens=8192,
    temperature=0.2,
    top_p=0.8,
    top_k=1,
    verbose=DEBUG,
)

prompt = PromptTemplate.from_template(get_prompt_template())

# Create a chain for passing a list of Documents to a model.
# The input is a dictionary that must have a “context” key that maps to a List[Document], and any other input variables expected in the prompt.
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

faissdb = create_vector_db()
retriever = faissdb.as_retriever()

# Create retrieval chain that retrieves documents and then passes them on.
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Flask section

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods= ['POST'])
def predict():
    if request.get_json():
        x=json.dumps(request.get_json())
        print('ok')
        x=json.loads(x)
    else:
        x={}
    data=x["text"]  # text

    result = retrieval_chain.invoke({"input": data})
    source_documents = list({doc.metadata["source"] for doc in result["context"]})

    response = {
        'answer': result["answer"],
        'sources': source_documents
    }

    response=jsonify(response)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    app.run(port=8080, host='0.0.0.0', debug=DEBUG)
