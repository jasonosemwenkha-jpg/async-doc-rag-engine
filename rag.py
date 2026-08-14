from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Fast local embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to vector store
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Fetch top 2 context chunks
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

def answer_query(question: str):
    docs = retriever.invoke(question)
    
    # Clean up escaped newlines and bad spacing
    cleaned_chunks = []
    for doc in docs:
        text = doc.page_content.replace(r"\n", "\n").replace("\n", " ")
        cleaned_chunks.append(text)
    
    context = "\n\n".join(cleaned_chunks)
    
    if not context:
        return "No relevant context found in the document database."
        
    return context