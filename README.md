#  Personal Research Assistant Agent

This project is an AI-powered assistant that allows users to upload multiple PDF research documents and receive accurate, context-based answers to their natural language questions. It combines PDF processing, semantic search, and large language models to make document understanding easy and interactive.



## Features

- Upload and analyze multiple PDF files
- Extract and split text into manageable chunks
- Generate semantic embeddings using `all-MiniLM-L6-v2`
- Store and retrieve content using FAISS vector store
- Answer questions using `google/flan-t5-base` (Hugging Face Transformers)
- Built with a simple and user-friendly Streamlit interface



## Tech Stack

- Python
- Streamlit
- PyPDF2
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- FAISS (Facebook AI Similarity Search)

## 📁 Files Included

- project.py → Core logic: loading PDFs, chunking, vector store, QA pipeline  
- app.py → Streamlit frontend for user interaction    
- dataset → Folder containing your PDF files


##  How to Run

1. **Clone the repository**
   - bash
   git clone https://github.com/yourusername/personal-research-assistant.git
   cd personal-research-assistant
