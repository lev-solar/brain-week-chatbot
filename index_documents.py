import os
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredHTMLLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class LocalEmbedding(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def load_docs():
    loaders = [
        DirectoryLoader("runbooks", glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader("runbooks", glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader("runbooks", glob="**/*.pdf", loader_cls=PyMuPDFLoader),
        DirectoryLoader("runbooks", glob="**/*.html", loader_cls=UnstructuredHTMLLoader)
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def main():
    docs = load_docs()
    # This code creates a text splitter and uses it to break down large documents into smaller, 
    # manageable chunks for processing. The RecursiveCharacterTextSplitter is a sophisticated 
    # text splitting algorithm from LangChain that intelligally divides documents while trying 
    # to preserve semantic meaning and context.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Split into {len(chunks)} chunks")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    embedder = LocalEmbedding()
    # generates numerical vector representations (embeddings) for all the text chunks
    # It uses a SentenceTransformer model (specifically "all-MiniLM-L6-v2") to convert 
    # each text string into a high-dimensional vector that captures its semantic meaning
    
    # The embeddings that are returned represent each text chunk as a dense vector in a multi-dimensional space, 
    # where semantically similar texts will have vectors that are close to each other. 
    # For example, texts about "database connection issues" and "database connectivity problems" would have similar 
    # embedding vectors even though they use different words, because the model understands their semantic relationship.
    embeddings = embedder.embed_documents(texts)

    # creates a list of tuples by pairing each text string with its corresponding embedding vector
    # Combine into (text, embedding) tuples
    text_embeddings = list(zip(texts, embeddings))

    # Create a FAISS (Facebook AI Similarity Search) index from pre-computed text embeddings.
    # FAISS is a powerful library designed for efficient similarity search and clustering of dense vectors,
    # making it ideal for applications like semantic search, recommendation systems, and document retrieval.
    db = FAISS.from_embeddings(text_embeddings, embedder)

    db.save_local("index")
    print("Index saved.")

if __name__ == "__main__":
    main()