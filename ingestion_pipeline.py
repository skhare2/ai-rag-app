import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader # Help us read files 
from langchain_text_splitters import RecursiveCharacterTextSplitter # Allows you to split the text for chunking
from langchain_openai import OpenAIEmbeddings # Embedding model to convert chunks to embeddings
from langchain_chroma import Chroma # We are using the chroma vector database
from dotenv import load_dotenv 


load_dotenv() #  Loading environment variables - so the key in the .env file

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}. Please add your company documents.")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks with overlap"""
   
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],  # Multiple separators
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = recursive_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    return vectorstore


def main():
    print("Main Method")
    # 1 - Loading the files
    documents = load_documents(docs_path="policies")

    # 2 - Chunking the files
    chunks = split_documents(documents)

    #3 - Embedding and Storing in a Vector DB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__": # Checks if the program is being run directly
    main()