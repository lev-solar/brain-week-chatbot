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

def build_chain():
    # Load FAISS Vector Index (Facebook AI Similarity Search)
    # Safe to allow_dangerous_deserialization since index was generated locally and not from an untrusted source
    db = FAISS.load_local("index", LocalEmbedding(), allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    llm = OllamaLLM(model="mistral")  # Or llama3, phi3, etc.

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # [User Query] → [Retriever] → [Top-matching text chunks from FAISS]
    #                             ↓
    #                      [LLM (Ollama)]
    #                             ↓
    #                   [Answer using documents]
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return qa_chain