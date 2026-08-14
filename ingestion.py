from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def process_pdf(file_path: str):
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Convert to embeddings and store in ChromaDB
    embeddings =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    return len(chunks)