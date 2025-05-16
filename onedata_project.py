
# Install Required Libraries

!pip install PyPDF2
!pip install langchain
!pip install langchain-community
!pip install faiss-cpu
!pip install transformers
!pip install torch

# Import Libraries

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.docstore.document import Document

# Function to Load One PDF

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# Function to Create Documents with Source Metadata

def create_documents_with_source(text, source_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": source_name}) for chunk in chunks]

# Function to Create Vectorstore from Documents

def create_vectorstore_from_docs(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Free & local
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Function to Create QA Chain

def create_qa_chain(vectorstore):
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain

# Load PDFs and Create Documents with Metadata

text_t20 = load_pdf("/content/ICC Men’s T20 World Cup 2024_ Comprehensive Report.pdf")
text_ipl = load_pdf("/content/IPL_2024_Comprehensive_Report.pdf")

docs_t20 = create_documents_with_source(text_t20, "T20_WC_2024")
docs_ipl = create_documents_with_source(text_ipl, "IPL_2024")


all_docs = docs_t20 + docs_ipl

# Create Vectorstore with Source Info

vectorstore = create_vectorstore_from_docs(all_docs)

# Create QA Chain

qa_chain = create_qa_chain(vectorstore)

# Ask Question

question = input("Enter your research question: ")
answer = qa_chain.invoke({"query": question})

print("\nQuestion:", question)
print("Answer:", answer["result"])

