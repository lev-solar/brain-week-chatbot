from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer

# Chat Logic with Memory
class LocalEmbedding(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def build_chain(model_name="llama3.1"):
    # Load FAISS Vector Index (Facebook AI Similarity Search)
    # Safe to allow_dangerous_deserialization since index was generated locally and not from an untrusted source
    db = FAISS.load_local("index", LocalEmbedding(), allow_dangerous_deserialization=True)
    
    # Converts the FAISS database object into a retriever interface, which provides a standardized way 
    # to search and retrieve relevant documents from the indexed collection.
    retriever = db.as_retriever()

    # Creates a large language model (LLM) instance using Ollama, specifically loading the "mistral" model. 
    # Ollama is a local LLM runtime that allows you to run various open-source language models directly on your machine 
    # without needing to make API calls to external services like OpenAI or Anthropic.
    llm = OllamaLLM(model=model_name) 

    # Creates a conversation memory object that enables your chatbot to maintain context across multiple interactions 
    # within a conversation session.
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # [User Query] → [Retriever] → [Top-matching text chunks from FAISS]
    #                             ↓
    #                      [LLM (Ollama)]
    #                             ↓
    #                   [Answer using documents]

    # Creates a conversational retrieval chain that combines all the components you've set up into a complete 
    # RAG (Retrieval-Augmented Generation) system.
    # The ConversationalRetrievalChain is a high-level LangChain construct that orchestrates the entire question-answering 
    # workflow by connecting your language model, document retriever, and conversation memory into a single, cohesive pipeline.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return qa_chain