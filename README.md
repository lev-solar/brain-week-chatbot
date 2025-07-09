# 🧠 Runbook Chatbot (Local)

A local chatbot that helps you investigate issues using **runbooks and troubleshooting documents** in formats like **PDF, HTML, Markdown, and plain text**. Powered by **local LLMs via Ollama**, **local embeddings via SentenceTransformers**, and **FAISS** for retrieval. Comes with a simple **Streamlit single-page UI**.

---

## ✨ Features

- ✅ Supports PDF, HTML, Markdown, and text files
- ✅ Embedding-based retrieval using `sentence-transformers`
- ✅ Fully local: no OpenAI API key required
- ✅ Uses Ollama to run LLMs like `mistral`, `llama3`, or `phi3`
- ✅ Memory support for multi-turn chat
- ✅ Streamlit single-page web app UI

---

## 🚀 Quickstart

### 1. 📥 Clone the Repo

```bash
git clone https://github.com/lev-solar/brain-week-chatbot.git
cd runbook-chatbot-local
```

### 2. 📦 Install Python Requirements

```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux

pip install -r requirements.txt
```

### 3. 🧠 Install Ollama (Local LLM Engine)

```bash
brew install ollama
```

Then start the Ollama server:

```bash
ollama serve
```

### 4. 🤖 Pull a Local Model

```bash
ollama run mistral
```

You can substitute mistral with:
llama3
phi3
gemma, etc.

### 5. 📂 Add Your Runbooks

```
Add documents to the /runbooks folder.

Supported formats:

.pdf – exported SOPs or incident reports
.html – web dashboards, docs
.md – GitHub or internal documentation
.txt – plain text logs or guides

You may organize them into subfolders.
```

### 6. 🏗️ Build the Vector Index

After adding documents, run:

```bash
python index_documents.py
```

```
This script:
Loads documents from the runbooks/ folder
Splits them into chunks (~500 characters)
Embeds them using all-MiniLM-L6-v2
Stores them in a FAISS index inside index/
```

### 7. 💬 Launch the Chat UI

Run the Streamlit web app:

```bash
streamlit run streamlit_app.py
```

### 🛠 File Overview

```
File / Folder	Description
index_documents.py	Indexes PDF/HTML/MD/TXT using sentence embeddings + FAISS
app.py	Loads FAISS index, connects to Ollama LLM, handles memory
streamlit_app.py	Streamlit frontend for querying the chatbot
runbooks/	Your documentation folder (PDF, HTML, TXT, MD)
index/	Generated FAISS vector index
requirements.txt	Python package dependencies
```

```
[User Query] → [Retriever] → [Top-matching text chunks from FAISS]
                                 ↓
                          [LLM (Ollama)]
                                 ↓
                       [Answer using documents]
```
