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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Split into {len(chunks)} chunks")

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    embedder = LocalEmbedding()
    embeddings = embedder.embed_documents(texts)

    # Combine into (text, embedding) tuples
    text_embeddings = list(zip(texts, embeddings))

    # Now correctly call FAISS.from_embeddings
    db = FAISS.from_embeddings(text_embeddings, embedder)

    db.save_local("index")
    print("Index saved.")

if __name__ == "__main__":
    main()